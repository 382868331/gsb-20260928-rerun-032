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

from .model import Costs, InterruptSource, ModeSpec, TaskSpec


# ---------------------------------------------------------------- 运行态

@dataclass
class Job:
    """任务的一次释放（作业）。record 为开放扩展字典（功耗采样等）。"""

    jid: int
    task: TaskSpec
    release: int
    abs_deadline: int
    period: int = 0
    rel_deadline: int = 0
    mode: str = "default"
    pc: int = 0                      # 当前程序步下标
    step_left: int = 0               # 当前 run 步剩余拍数
    state: str = "ready"             # ready/running/blocked/done
    held: List[str] = field(default_factory=list)
    blocked_on: Optional[str] = None
    lock_request_mode: Optional[str] = None
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
    mode: Optional[str] = None


@dataclass
class MissRecord:
    jid: int
    task: str
    release: int
    deadline: int
    remaining_work: int
    release_mode: str = "default"
    deadline_mode: str = "default"


@dataclass
class ModeEvent:
    time: int
    mode: str
    strategy: str       # immediate / after_job
    requested_at: Optional[int] = None
    target_jid: Optional[int] = None


@dataclass
class LockBudgetViolation:
    resource: str
    jid: int
    task: str
    mode: str
    acquired_at: int
    budget: int
    first_violation_at: int
    elapsed_at_violation: int
    overrun: int
    released_at: Optional[int] = None


class System:
    def __init__(
        self,
        tasks: List[TaskSpec],
        interrupts: Optional[List[InterruptSource]] = None,
        costs: Costs = Costs(),
        use_inheritance: bool = True,
        modes: Optional[List[ModeSpec]] = None,
        initial_mode: Optional[str] = None,
    ):
        self.tasks = {t.name: t for t in tasks}
        self.interrupts = interrupts or []
        self.costs = costs
        self.use_inheritance = use_inheritance
        self.resources: Dict[str, Dict] = {}
        self.modes: Dict[str, ModeSpec] = {}
        if modes:
            for mode in modes:
                self.modes[mode.name] = mode
        self.mode = initial_mode or ("default" if not modes else modes[0].name)
        if modes and self.mode not in self.modes:
            raise KeyError("unknown initial mode: %r" % self.mode)
        self.mode_events: List[ModeEvent] = [
            ModeEvent(0, self.mode, "initial")
        ]
        self.pending_mode_change: Optional[ModeEvent] = None
        self.lock_budget_violations: List[LockBudgetViolation] = []

        self.now = 0
        self.jobs: List[Job] = []
        self._next_jid = 0
        self.current: Optional[Job] = None
        self._switch_to: Optional[Job] = None
        self._switch_left = 0
        self._isr_stack: List[Dict] = []
        self._irq_pending: List[Dict] = []
        self._next_release = {t.name: t.offset for t in tasks}
        self._irq_schedule = sorted(
            (time, i, src)
            for i, src in enumerate(self.interrupts)
            for time in src.times
        )
        self.trace: List[Segment] = []
        self.misses: List[MissRecord] = []

    def _mode_spec(self, mode_name: Optional[str] = None) -> Optional[ModeSpec]:
        return self.modes.get(mode_name or self.mode)

    def _task_period(self, task_name: str, static_period: int) -> int:
        mode = self._mode_spec()
        if mode is None:
            return static_period
        value = mode.periods.get(task_name, static_period)
        return static_period if value is None else value

    def _task_rel_deadline(self, task_name: str, task: TaskSpec) -> int:
        mode = self._mode_spec()
        if mode is None:
            return task.rel_deadline
        if task_name in mode.deadlines:
            value = mode.deadlines[task_name]
            if value is not None:
                return value
        period = self._task_period(task_name, task.period)
        return period if task.deadline is None else task.deadline

    # ------------------------------------------------------------ 释放与事件

    def _release_due(self):
        for t in self.tasks.values():
            if self.now == self._next_release[t.name]:
                period = self._task_period(t.name, t.period)
                rel_deadline = self._task_rel_deadline(t.name, t)
                release_mode = self.mode
                self._next_release[t.name] += period
                job = Job(
                    jid=self._next_jid,
                    task=t,
                    release=self.now,
                    abs_deadline=self.now + rel_deadline,
                    period=period,
                    rel_deadline=rel_deadline,
                    mode=release_mode,
                )
                if t.program and t.program[0].kind == "run":
                    job.step_left = t.program[0].arg
                self._next_jid += 1
                self.jobs.append(job)
        while self._irq_schedule and self._irq_schedule[0][0] <= self.now:
            _, _, src = self._irq_schedule.pop(0)
            self._irq_pending.append({"src": src, "left": src.total_cost})

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
            res_name,
            {
                "holder": None,
                "waiters": [],
                "acquired_at": None,
                "name": res_name,
                "mode": None,
                "budget": None,
                "violation": None,
            },
        )
        if res["holder"] is None:
            res["holder"] = job
            job.held.append(res_name)
            request_mode = job.lock_request_mode or self.mode
            res["acquired_at"] = self.now + 1
            res["mode"] = request_mode
            mode_spec = self._mode_spec(request_mode)
            res["budget"] = (
                mode_spec.lock_budgets.get(res_name)
                if mode_spec is not None else None
            )
            job.lock_request_mode = None
            job.pc += 1
            self._prime_step(job)
        else:
            job.state = "blocked"
            job.blocked_on = res_name
            if job.lock_request_mode is None:
                job.lock_request_mode = self.mode
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
            self._check_lock_budgets(self.now + 1)
            self._finish_lock_budget(res, self.now + 1)
            res["acquired_at"] = None
            res["mode"] = None
            res["budget"] = None
            # 等待者回到就绪态，被调度到时重新执行 lock 系统调用竞争锁；
            # 未抢到者再次阻塞，其继承压力随之重新挂到新持有者上。
            for w in list(res["waiters"]):
                w.state = "ready"
                w.blocked_on = None
            res["waiters"].clear()
        job.pc += 1
        self._prime_step(job)
        self._recompute_boosts()

    def _check_lock_budgets(self, at_time: Optional[int] = None):
        at_time = self.now if at_time is None else at_time
        for res in self.resources.values():
            if res["holder"] is None or res["budget"] is None:
                continue
            if res["violation"] is not None:
                continue
            acquired_at = res["acquired_at"]
            elapsed = at_time - acquired_at
            if elapsed > res["budget"]:
                holder = res["holder"]
                violation = LockBudgetViolation(
                    resource=res["name"],
                    jid=holder.jid,
                    task=holder.task.name,
                    mode=res["mode"],
                    acquired_at=acquired_at,
                    budget=res["budget"],
                    first_violation_at=at_time,
                    elapsed_at_violation=elapsed,
                    overrun=elapsed - res["budget"],
                )
                res["violation"] = violation
                self.lock_budget_violations.append(violation)

    def _finish_lock_budget(self, res: Dict, end_time: int):
        violation = res.get("violation")
        if violation is not None and violation.released_at is None:
            violation.released_at = end_time

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

    # ------------------------------------------------------------ 模式切换

    def switch_mode(
        self,
        mode_name: str,
        strategy: str = "immediate",
        replace_pending: bool = True,
        at_time: Optional[int] = None,
    ):
        """在当前时间请求模式切换。

        immediate 默认在下一拍边界生效；给出 at_time 时在指定边界生效。
        after_job 等当前运行作业完成后生效。
        请求本身不占用 CPU，也不回滚任何作业的执行进度或截止承诺。
        """
        if strategy not in ("immediate", "after_job"):
            raise ValueError("unknown mode switch strategy: %r" % strategy)
        if mode_name not in self.modes:
            raise KeyError("unknown mode: %r" % mode_name)
        if at_time is not None and at_time < self.now:
            raise ValueError("mode switch time cannot be in the past")
        target = self._current_worker_job() if strategy == "after_job" else None
        event = ModeEvent(
            time=self.now if at_time is None else at_time,
            mode=mode_name,
            strategy=strategy,
            requested_at=self.now,
            target_jid=target.jid if target is not None else None,
        )
        if strategy == "immediate":
            if replace_pending or self.pending_mode_change is None:
                self.pending_mode_change = event
        elif self.pending_mode_change is None or replace_pending:
            self.pending_mode_change = event

    def _apply_due_mode_change(self, boundary: Optional[int] = None):
        event = self.pending_mode_change
        if event is None:
            return
        if event.strategy == "immediate":
            due = event.time <= self.now
        elif event.target_jid is None:
            due = True
        else:
            target = next(
                (j for j in self.jobs if j.jid == event.target_jid), None
            )
            due = target is None or target.state == "done"
        if not due:
            return
        event.time = self.now if boundary is None else boundary
        self.mode = event.mode
        self.mode_events.append(event)
        self.pending_mode_change = None

    def _current_worker_job(self) -> Optional[Job]:
        if self.current is not None and self.current.state != "done":
            return self.current
        for segment in reversed(self.trace):
            if segment.jid is None:
                continue
            job = next((j for j in self.jobs if j.jid == segment.jid), None)
            if job is not None and job.state != "done":
                return job
        return None

    # ------------------------------------------------------------ 记录

    def _emit(self, kind: str, detail: str, jid: Optional[int] = None):
        if self.trace and self.trace[-1].end == self.now \
                and self.trace[-1].kind == kind \
                and self.trace[-1].detail == detail \
                and self.trace[-1].jid == jid \
                and self.trace[-1].mode == self.mode:
            self.trace[-1].end = self.now + 1
        else:
            self.trace.append(
                Segment(self.now, self.now + 1, kind, detail, jid, self.mode)
            )

    # ------------------------------------------------------------ 主循环

    def step(self):
        """推进一拍（切换决策本身不耗拍，切换开销在后续拍计入）。"""
        self._apply_due_mode_change()
        self._release_due()

        if self._irq_pending:
            top = self._isr_stack[-1] if self._isr_stack else None
            top_priority = -10**18 if top is None else top["src"].priority
            best_index = max(
                range(len(self._irq_pending)),
                key=lambda i: self._irq_pending[i]["src"].priority,
            )
            if self._irq_pending[best_index]["src"].priority > top_priority:
                self._isr_stack.append(self._irq_pending.pop(best_index))

        if self._isr_stack:
            frame = self._isr_stack[-1]
            src = frame["src"]
            self._emit("isr", "isr:" + src.name)
            frame["left"] -= 1
            if frame["left"] == 0:
                self._isr_stack.pop()
            self._finish_tick()
            self._check_lock_budgets()
            return

        if self._switch_left > 0:
            self._emit("switch", "switch->" + self._label(self._switch_to))
            self._switch_left -= 1
            if self._switch_left == 0:
                self._activate(self._switch_to)
                self._switch_to = None
            self._finish_tick()
            self._check_lock_budgets()
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
            self._check_lock_budgets()
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
            if sel.lock_request_mode is None:
                sel.lock_request_mode = self.mode
            self._try_lock(sel, step.arg)
        elif step.kind == "unlock":
            self._emit("unlock", label + " unlock " + str(step.arg), sel.jid)
            self._do_unlock(sel, step.arg)
        self._finish_tick()
        self._check_lock_budgets()

    def _activate(self, job: Optional[Job]):
        self.current = job
        if job is not None:
            job.state = "running"

    def _finish_tick(self):
        self.now += 1
        self._apply_due_mode_change(self.now)
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
                    release_mode=job.mode,
                    deadline_mode=self.mode,
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
