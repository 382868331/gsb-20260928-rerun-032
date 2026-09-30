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

from .model import Costs, InterruptSource, TaskSpec


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


@dataclass
class MissRecord:
    jid: int
    task: str
    release: int
    deadline: int
    remaining_work: int


class System:
    def __init__(
        self,
        tasks: List[TaskSpec],
        interrupts: Optional[List[InterruptSource]] = None,
        costs: Costs = Costs(),
        use_inheritance: bool = True,
    ):
        self.tasks = {t.name: t for t in tasks}
        self.interrupts = interrupts or []
        self.costs = costs
        self.use_inheritance = use_inheritance
        self.resources: Dict[str, Dict] = {}

        self.now = 0
        self.jobs: List[Job] = []
        self._next_jid = 0
        self.current: Optional[Job] = None
        self._switch_to: Optional[Job] = None
        self._switch_left = 0
        self._isr: Optional[Dict] = None   # {name, left}
        self._irq_pending: List[InterruptSource] = []
        self._next_release = {t.name: t.offset for t in tasks}
        self._irq_schedule = sorted(
            (time, i, src)
            for i, src in enumerate(self.interrupts)
            for time in src.times
        )
        self.trace: List[Segment] = []
        self.misses: List[MissRecord] = []

    # ------------------------------------------------------------ 释放与事件

    def _release_due(self):
        for t in self.tasks.values():
            if self.now == self._next_release[t.name]:
                self._next_release[t.name] += t.period
                job = Job(
                    jid=self._next_jid,
                    task=t,
                    release=self.now,
                    abs_deadline=self.now + t.rel_deadline,
                )
                if t.program and t.program[0].kind == "run":
                    job.step_left = t.program[0].arg
                self._next_jid += 1
                self.jobs.append(job)
        while self._irq_schedule and self._irq_schedule[0][0] <= self.now:
            _, _, src = self._irq_schedule.pop(0)
            self._irq_pending.append(src)

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
        if self.trace and self.trace[-1].end == self.now \
                and self.trace[-1].kind == kind \
                and self.trace[-1].detail == detail:
            self.trace[-1].end = self.now + 1
        else:
            self.trace.append(Segment(self.now, self.now + 1, kind, detail, jid))

    # ------------------------------------------------------------ 主循环

    def step(self):
        """推进一拍（切换决策本身不耗拍，切换开销在后续拍计入）。"""
        self._release_due()

        # 中断触发：立即抢占（进入/退出开销已含在 ISR 总耗时内）
        if self._isr is None and self._irq_pending:
            src = self._irq_pending.pop(0)
            self._isr = {"name": src.name, "left": src.total_cost}

        if self._isr is not None:
            self._emit("isr", "isr:" + self._isr["name"])
            self._isr["left"] -= 1
            if self._isr["left"] == 0:
                self._isr = None
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
                ))

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
