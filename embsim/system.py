"""虚拟处理器与可抢占调度器（逐拍推进，产生逐段执行记录）。

时间模型
--------
- 时间为整数“拍”。每一拍 CPU 只做一件事：跑某个任务、跑 ISR、
  做上下文切换、执行 lock/unlock 系统调用，或空转。
- 抢占、阻塞、中断的代价都体现为 CPU 时间上的段，
  因此错过截止时间一定能在执行记录里找到对应的开销段。

优先级反转处理
--------------
采用立即优先级继承（immediate / basic priority inheritance）：
任务 J 阻塞在资源 R 上时，R 的持有者优先级立即提升为
所有等待者（含传递等待）的最高有效优先级；释放 R 后恢复。
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .model import Costs, InterruptSource, Mode, TaskSpec


# ---------------------------------------------------------------- 运行态

@dataclass
class Job:
    """任务的一次释放（作业）。record 为开放扩展字典（功耗采样等）。"""

    jid: int
    task: TaskSpec
    release: int
    abs_deadline: int
    pc: int = 0                      # 当前程序步下标
    step_left: int = 0               # 当前 run 步剩余拍数
    state: str = "ready"             # ready/running/blocked/done
    held: List[str] = field(default_factory=list)
    blocked_on: Optional[str] = None
    boost: int = 0                   # 继承得到的优先级增量
    completed_at: Optional[int] = None
    missed: bool = False
    release_mode: Optional[str] = None   # 释放时所处模式
    deadline_mode: Optional[str] = None  # 承诺截止时刻所依据的模式
    # 每把锁在各模式下的实际持有拍数：{资源: {模式: 拍数}}
    lock_held_by_mode: Dict[str, Dict[str, int]] = field(default_factory=dict)
    record: Dict = field(default_factory=dict)

    @property
    def base_priority(self) -> int:
        return self.task.priority

    @property
    def eff_priority(self) -> int:
        return self.task.priority + self.boost

    def remaining_work(self) -> int:
        left = 0
        for i, s in enumerate(self.task.program):
            if s.kind != "run":
                continue
            if i > self.pc:
                left += s.arg
            elif i == self.pc:
                left += self.step_left
        return left


@dataclass
class Segment:
    """逐段执行记录的一段：[start, end) 上 CPU 在做什么。"""

    start: int
    end: int
    kind: str      # task / lock / unlock / switch / isr / idle
    detail: str    # 如 "T1#0"、"isr:Timer"、"switch->T2#1"
    jid: Optional[int] = None
    mode: str = ""  # 该拍所处工作模式（"" 表示未启用模式）


@dataclass
class MissRecord:
    jid: int
    task: str
    release: int
    deadline: int
    remaining_work: int
    deadline_mode: Optional[str] = None  # 截止承诺归属的模式（冻结）


@dataclass
class ModeChange:
    """一次工作模式切换（不可变历史，保存/恢复后必须逐拍重现）。"""

    time: int
    old_mode: Optional[str]
    new_mode: Optional[str]
    strategy: str  # immediate / wait_job / scheduled


@dataclass
class BudgetViolation:
    """某模式下某把锁的持有占用超出该模式允许预算。

    记账按“持锁拍发生时的当前模式”归类，因此跨模式持锁会在新模式下
    继续累计并可能告警；历史告警不会因之后放锁或再次切换而被改写。
    """

    jid: int
    resource: str
    mode: Optional[str]
    time: int                 # 首次越界的拍末时刻
    held_in_mode: int         # 截止该拍在该模式下已持有拍数
    budget: int               # 该模式允许预算


class System:
    def __init__(
        self,
        tasks: List[TaskSpec],
        interrupts: Optional[List[InterruptSource]] = None,
        costs: Costs = Costs(),
        use_inheritance: bool = True,
        modes: Optional[List[Mode]] = None,
        initial_mode: Optional[str] = None,
    ):
        self.tasks = {t.name: t for t in tasks}
        self.interrupts = interrupts or []
        self.costs = costs
        self.use_inheritance = use_inheritance
        self.modes: Dict[str, Mode] = {m.name: m for m in (modes or [])}
        if initial_mode is not None:
            assert initial_mode in self.modes, "未知模式: %r" % initial_mode
        self.mode_name: Optional[str] = initial_mode
        self.resources: Dict[str, Dict] = {}

        self.now = 0
        self.jobs: List[Job] = []
        self._next_jid = 0
        self.current: Optional[Job] = None
        self._switch_to: Optional[Job] = None
        self._switch_left = 0
        self._isr_stack: List[Dict] = []    # ISR 嵌套栈：栈顶为正在服务者
        self._irq_pending: List[InterruptSource] = []
        self._next_release = {t.name: t.offset for t in tasks}
        self._irq_schedule = sorted(
            (time, i, src)
            for i, src in enumerate(self.interrupts)
            for time in src.times
        )
        self.trace: List[Segment] = []
        self.misses: List[MissRecord] = []
        self.mode_changes: List[ModeChange] = []
        self.budget_violations: List[BudgetViolation] = []
        # 已记录越界的 (jid, resource, mode)，同一组合只在首次越界告警
        self._budget_flagged: set = set()
        # 模式切换请求：[(time, mode_name)]，到点立即切换
        self._pending_switches: List = []
        # “等作业结束”请求：[(mode_name, 登记时已存在的作业 jid 集合)]
        self._pending_wait_switches: List = []

    # ------------------------------------------------------------ 模式

    @property
    def mode(self) -> Optional[Mode]:
        if self.mode_name is None:
            return None
        return self.modes[self.mode_name]

    def _mode_label(self) -> str:
        return self.mode_name or ""

    def switch_mode(self, mode_name: Optional[str]) -> None:
        """立即切换模式：在当前时刻（拍边界）生效。

        已释放作业的周期/截止承诺不变，只调整未释放作业的后续释放表；
        正在执行/阻塞的作业继续沿用各自程序步与锁状态。
        """
        if mode_name is not None and mode_name not in self.modes:
            raise KeyError("未知模式: %r" % mode_name)
        if mode_name == self.mode_name:
            return
        old = self.mode_name
        self.mode_name = mode_name
        self.mode_changes.append(
            ModeChange(self.now, old, mode_name, "immediate")
        )
        self._reschedule_after_switch()

    def switch_mode_after_jobs(self, mode_name: Optional[str]) -> None:
        """登记“等当前作业结束后切换”：请求时已存在的未完成作业全部
        done 后自动切换；等待期间新释放的作业不延长等待。"""
        if mode_name is not None and mode_name not in self.modes:
            raise KeyError("未知模式: %r" % mode_name)
        witness = {j.jid for j in self.jobs if j.state != "done"}
        self._pending_wait_switches.append((mode_name, witness))

    def schedule_mode_switch(self, time: int, mode_name: Optional[str]) -> None:
        """登记在指定时刻立即切换（用于对比/编排，含恰好落在截止拍）。"""
        if mode_name is not None and mode_name not in self.modes:
            raise KeyError("未知模式: %r" % mode_name)
        self._pending_switches.append((time, mode_name))
        self._pending_switches.sort(key=lambda x: x[0])

    def _reschedule_after_switch(self):
        """新周期从切换时刻起对“尚未释放”的作业生效。

        已经释放（活在作业列表里）的作业承诺一律不动；尚未到点的下一次
        释放是新模式时间线的一部分，重排为 now + 新模式周期，
        从而保证新模式周期/截止参数对新作业立即生效、单一可追溯。
        """
        for t in self.tasks.values():
            nxt = self._next_release[t.name]
            if nxt > self.now:
                self._next_release[t.name] = self.now + self._period_of(t)

    def _period_of(self, task: TaskSpec) -> int:
        m = self.mode
        return task.period if m is None else m.period_of(task)

    def _deadline_of(self, task: TaskSpec) -> int:
        m = self.mode
        return task.rel_deadline if m is None else m.deadline_of(task)

    # ------------------------------------------------------------ 释放与事件

    def _release_due(self):
        # 1) 到点的预定模式切换（先于本拍释放，使“该拍属于哪个模式”唯一）
        while self._pending_switches \
                and self._pending_switches[0][0] <= self.now:
            _, target = self._pending_switches.pop(0)
            if target != self.mode_name:
                old = self.mode_name
                self.mode_name = target
                self.mode_changes.append(
                    ModeChange(self.now, old, target, "scheduled")
                )
                self._reschedule_after_switch()
        # 2) 本拍到期的作业释放；释放时刻也在本拍内确定（同一拍内先释放，
        #    随后才检查“等作业结束”切换，保证已存在作业至少经历一拍）。
        for t in self.tasks.values():
            if self.now == self._next_release[t.name]:
                self._next_release[t.name] += self._period_of(t)
                job = Job(
                    jid=self._next_jid,
                    task=t,
                    release=self.now,
                    abs_deadline=self.now + self._deadline_of(t),
                )
                job.release_mode = self.mode_name
                job.deadline_mode = self.mode_name
                if t.program and t.program[0].kind == "run":
                    job.step_left = t.program[0].arg
                self._next_jid += 1
                self.jobs.append(job)
        while self._irq_schedule and self._irq_schedule[0][0] <= self.now:
            _, _, src = self._irq_schedule.pop(0)
            self._irq_pending.append(src)

    def _apply_wait_switches(self):
        """拍末检查：登记时已存在的作业全部完成，则切换到等待模式。"""
        while self._pending_wait_switches:
            target, witness = self._pending_wait_switches[0]
            live = {j.jid: j for j in self.jobs}
            if any(live[jid].state != "done"
                   for jid in witness if jid in live):
                return
            self._pending_wait_switches.pop(0)
            if target != self.mode_name:
                old = self.mode_name
                self.mode_name = target
                self.mode_changes.append(
                    ModeChange(self.now, old, target, "wait_job")
                )
                self._reschedule_after_switch()

    # ------------------------------------------------------------ 优先级继承

    def _recompute_boosts(self):
        if not self.use_inheritance:
            return
        for _ in range(len(self.jobs) + 1):  # 迭代到不动点，覆盖传递阻塞
            changed = False
            for job in self.jobs:
                if job.state == "done":
                    continue
                want = 0
                for res_name in job.held:
                    res = self.resources[res_name]
                    for w in res["waiters"]:
                        want = max(want, w.eff_priority - job.base_priority)
                want = max(0, want)
                if want != job.boost:
                    job.boost = want
                    changed = True
            if not changed:
                break

    # ------------------------------------------------------------ 资源操作

    def _try_lock(self, job: Job, res_name: str):
        res = self.resources.setdefault(
            res_name, {"holder": None, "waiters": []}
        )
        if res["holder"] is None:
            res["holder"] = job
            job.held.append(res_name)
            job.pc += 1
            self._prime_step(job)
        else:
            job.state = "blocked"
            job.blocked_on = res_name
            if job not in res["waiters"]:
                res["waiters"].append(job)
            if self.current is job:
                self.current = None
        self._recompute_boosts()

    def _do_unlock(self, job: Job, res_name: str):
        res = self.resources[res_name]
        if res["holder"] is job:
            res["holder"] = None
            job.held.remove(res_name)
            # 等待者回到就绪态，被调度到时重新执行 lock 系统调用竞争锁；
            # 未抢到者再次阻塞，其继承压力随之重新挂到新持有者上。
            for w in list(res["waiters"]):
                w.state = "ready"
                w.blocked_on = None
            res["waiters"].clear()
        job.pc += 1
        self._prime_step(job)
        self._recompute_boosts()

    def _prime_step(self, job: Job):
        """推进到下一步；若程序结束则标记完成（完成时刻为当前拍末）。"""
        if job.pc < len(job.task.program):
            s = job.task.program[job.pc]
            if s.kind == "run":
                job.step_left = s.arg
        else:
            job.state = "done"
            job.completed_at = self.now + 1
            job.boost = 0  # 解除后恢复：完成作业不再向任何持有者传递优先级
            if self.current is job:
                self.current = None

    # ------------------------------------------------------------ 选择

    def _ready_jobs(self) -> List[Job]:
        return [j for j in self.jobs if j.state in ("ready", "running")]

    def _select(self) -> Optional[Job]:
        ready = self._ready_jobs()
        if not ready:
            return None
        return max(ready, key=lambda j: (j.eff_priority, -j.jid))

    # ------------------------------------------------------------ 记录

    def _emit(self, kind: str, detail: str, jid: Optional[int] = None):
        mode = self._mode_label()
        if self.trace and self.trace[-1].end == self.now \
                and self.trace[-1].kind == kind \
                and self.trace[-1].detail == detail \
                and self.trace[-1].mode == mode:
            self.trace[-1].end = self.now + 1
        else:
            self.trace.append(
                Segment(self.now, self.now + 1, kind, detail, jid, mode)
            )

    # ------------------------------------------------------------ 主循环

    def step(self):
        """推进一拍（切换决策本身不耗拍，切换开销在后续拍计入）。"""
        self._release_due()

        # 中断嵌套：未决中断里优先级最高者，若高于栈顶 ISR 则抢占入栈；
        # 同优先级按触发先后（_irq_pending 保序）。
        if self._irq_pending:
            top_pri = self._isr_stack[-1]["pri"] if self._isr_stack else None
            idx = 0
            best = 0
            for src in self._irq_pending:
                if top_pri is None or src.priority > top_pri:
                    best = idx
                    break
                idx += 1
            else:
                best = -1
            if best >= 0:
                src = self._irq_pending.pop(best)
                self._isr_stack.append(
                    {"name": src.name, "left": src.total_cost,
                     "pri": src.priority}
                )

        if self._isr_stack:
            isr = self._isr_stack[-1]
            self._emit("isr", "isr:" + isr["name"])
            isr["left"] -= 1
            if isr["left"] == 0:
                self._isr_stack.pop()
            self._finish_tick()
            return

        if self._switch_left > 0:
            self._emit("switch", "switch->" + self._label(self._switch_to))
            self._switch_left -= 1
            if self._switch_left == 0:
                self._activate(self._switch_to)
                self._switch_to = None
            self._finish_tick()
            return

        sel = self._select()
        if sel is not None and sel is not self.current:
            # 发生抢占/调度：先付上下文切换代价
            if self.current is not None and self.current.state == "running":
                self.current.state = "ready"
            self.current = None
            if self.costs.context_switch > 0:
                self._switch_to = sel
                self._switch_left = self.costs.context_switch
                return  # 决策不耗拍
            self._activate(sel)

        if sel is None:
            self._emit("idle", "idle")
            self._finish_tick()
            return

        # 执行当前作业一拍
        step = sel.task.program[sel.pc]
        label = self._label(sel)
        if step.kind == "run":
            self._emit("task", label, sel.jid)
            sel.step_left -= 1
            if sel.step_left == 0:
                sel.pc += 1
                self._prime_step(sel)
        elif step.kind == "lock":
            self._emit("lock", label + " lock " + str(step.arg), sel.jid)
            self._try_lock(sel, step.arg)
        elif step.kind == "unlock":
            self._emit("unlock", label + " unlock " + str(step.arg), sel.jid)
            self._do_unlock(sel, step.arg)
        self._finish_tick()

    def _activate(self, job: Optional[Job]):
        self.current = job
        if job is not None:
            job.state = "running"

    def _finish_tick(self):
        tick_mode = self.mode_name
        # 锁占用预算：按“持锁拍发生时的当前模式”记账。每把被持有的锁
        # 在每拍向持有者在该模式下累计 1 拍，越过预算即冻结一次告警。
        for res_name, res in self.resources.items():
            holder = res["holder"]
            if holder is None:
                continue
            per_mode = holder.lock_held_by_mode.setdefault(res_name, {})
            per_mode[tick_mode] = per_mode.get(tick_mode, 0) + 1
            m = self.mode
            budget = None if m is None else m.budget_of(res_name)
            held = per_mode[tick_mode]
            if budget is not None and held > budget:
                key = (holder.jid, res_name, tick_mode)
                if key not in self._budget_flagged:
                    self._budget_flagged.add(key)
                    self.budget_violations.append(BudgetViolation(
                        jid=holder.jid,
                        resource=res_name,
                        mode=tick_mode,
                        time=self.now + 1,
                        held_in_mode=held,
                        budget=budget,
                    ))
        self.now += 1
        for job in self.jobs:
            if (not job.missed and job.state != "done"
                    and self.now >= job.abs_deadline):
                job.missed = True
                self.misses.append(MissRecord(
                    jid=job.jid,
                    task=job.task.name,
                    release=job.release,
                    deadline=job.abs_deadline,
                    remaining_work=job.remaining_work(),
                    deadline_mode=job.deadline_mode,
                ))
        # 截止检查之后再处理“等作业结束”切换：恰好截止拍完成的作业
        # 在本拍仍属旧模式，切换只能从下一拍起生效。
        self._apply_wait_switches()

    @staticmethod
    def _label(job: Job) -> str:
        return "%s#%d" % (job.task.name, job.jid)

    def run(self, ticks: int):
        self.run_until(self.now + ticks)

    def run_until(self, t: int):
        guard = 0
        while self.now < t:
            self.step()
            guard += 1
            if guard > 10 ** 7:
                raise RuntimeError("仿真未收敛，疑似死循环")

    # ------------------------------------------------------------ 快照

    def snapshot(self) -> bytes:
        """序列化完整运行态（任务记录、作业、资源、追踪、时钟）。"""
        return pickle.dumps(self)

    @staticmethod
    def restore(data: bytes) -> "System":
        return pickle.loads(data)

    def save(self, path: str):
        with open(path, "wb") as f:
            f.write(self.snapshot())

    @staticmethod
    def load(path: str) -> "System":
        with open(path, "rb") as f:
            return System.restore(f.read())
