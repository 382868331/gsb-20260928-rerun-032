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
    esc_boost: int = 0               # 关键程度升级得到的优先级增量
    deferred: bool = False           # 是否因让步策略被推迟
    orig_deadline: int = 0           # 释放时承诺的绝对截止（升级可临时收紧）
    completed_at: Optional[int] = None
    missed: bool = False
    record: Dict = field(default_factory=dict)

    @property
    def base_priority(self) -> int:
        return self.task.priority

    @property
    def eff_priority(self) -> int:
        return self.task.priority + self.boost + self.esc_boost

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


@dataclass
class Escalation:
    """一次关键程度升级承诺。

    trigger: "time"（到达指定拍边界）/ "lock"（目标任务获得某资源）/
    "step"（目标任务程序计数到达某步）。触发后目标作业获得
    priority_raise 的优先级增量；若给出 response，其绝对截止被临时收紧为
    applied_at + response（不晚于原承诺截止）。strategy 决定其他低关键
    任务如何让步：
    - "priority": 只抬优先级，不额外推迟任何任务；
    - "defer":    有效优先级低于升级目标的作业被推迟，但持有资源的
                  作业必须先完成临界区（已占有的资源不能凭空消失），
                  放锁后才可暂停，保证锁最终一定被释放。
    state: pending -> active -> revoked/done，生效范围与先后关系全部
    保留在 escalations 列表里，不会被后来的升级或撤回改写。
    """

    eid: int
    task: str
    trigger: str
    trigger_arg: object
    priority_raise: int
    response: Optional[int]
    strategy: str
    state: str = "pending"
    requested_at: int = 0
    applied_at: Optional[int] = None
    ended_at: Optional[int] = None
    end_reason: Optional[str] = None
    target_jid: Optional[int] = None
    saved_deadline: Optional[int] = None


@dataclass
class DeferralRecord:
    """一次让步推迟记录：哪个作业、何时被推迟、何时恢复及原因。"""

    jid: int
    task: str
    eid: int
    deferred_at: int
    cause: str
    resumed_at: Optional[int] = None
    reason: Optional[str] = None


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
        self.escalations: List[Escalation] = []
        self.deferrals: List[DeferralRecord] = []
        self._next_eid = 0

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
                    orig_deadline=self.now + rel_deadline,
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
            self._check_escalation_triggers(job, ("lock", res_name))
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
            self._check_escalation_triggers(job, ("step", job.pc))
        else:
            job.state = "done"
            job.completed_at = self.now + 1
            job.boost = 0  # 解除后恢复：完成作业不再向任何持有者传递优先级
            self._end_escalations_of(job)
            self._recompute_escalation_effects()
            if self.current is job:
                self.current = None

    # ------------------------------------------------------------ 选择

    def _ready_jobs(self) -> List[Job]:
        return [j for j in self.jobs
                if j.state in ("ready", "running") and not j.deferred]

    # ------------------------------------------------------------ 关键程度升级

    def escalate(
        self,
        task_name: str,
        priority_raise: int,
        response: Optional[int] = None,
        strategy: str = "defer",
        at_time: Optional[int] = None,
        on_lock: Optional[str] = None,
        on_step: Optional[int] = None,
    ) -> int:
        """请求临时提高某任务的关键程度，返回升级编号 eid。

        触发事件三选一（缺省为 at_time=当前时刻，即下一拍边界生效）：
        at_time=t  到达指定拍边界；on_lock=R  目标任务获得资源 R；
        on_step=k  目标任务程序计数到达第 k 步。
        response 给出触发后更紧的响应要求（新的相对截止，按拍计）；
        strategy 为 "defer"（低关键任务让步推迟）或 "priority"（仅抬优先级）。
        请求本身不占 CPU，不回滚任何已消耗的执行时间或已占有的资源。
        """
        if task_name not in self.tasks:
            raise KeyError("unknown task: %r" % task_name)
        if strategy not in ("defer", "priority"):
            raise ValueError("unknown escalation strategy: %r" % strategy)
        triggers = [t is not None for t in (at_time, on_lock, on_step)]
        if sum(triggers) > 1:
            raise ValueError("escalation trigger must be unique")
        if on_lock is not None:
            trigger, arg = "lock", on_lock
        elif on_step is not None:
            trigger, arg = "step", on_step
        else:
            trigger, arg = "time", self.now if at_time is None else at_time
            if arg < self.now:
                raise ValueError("escalation time cannot be in the past")
        if response is not None and response <= 0:
            raise ValueError("response must be positive")
        event = Escalation(
            eid=self._next_eid,
            task=task_name,
            trigger=trigger,
            trigger_arg=arg,
            priority_raise=priority_raise,
            response=response,
            strategy=strategy,
            requested_at=self.now,
        )
        self._next_eid += 1
        self.escalations.append(event)
        return event.eid

    def revoke(self, eid: int):
        """撤回升级承诺：优先级增量与收紧的截止即刻（当前拍边界）解除，
        被推迟的作业恢复；已记录的超期事实保持冻结，不被改写。"""
        event = next((e for e in self.escalations if e.eid == eid), None)
        if event is None:
            raise KeyError("no escalation with eid=%r" % (eid,))
        if event.state not in ("pending", "active"):
            raise ValueError("escalation #%d already ended" % eid)
        event.state = "revoked"
        event.ended_at = self.now
        event.end_reason = "revoked"
        self._recompute_escalation_effects()
        self._update_deferrals()

    def _check_escalation_triggers(self, job=None, event=None):
        for esc in self.escalations:
            if esc.state != "pending":
                continue
            if esc.trigger == "time":
                if self.now < esc.trigger_arg:
                    continue
                target = self._live_job_of(esc.task)
                if target is not None:
                    self._apply_escalation(esc, target)
            elif esc.trigger == "lock" and job is not None:
                kind, res_name = event
                if job.task.name == esc.task and res_name == esc.trigger_arg:
                    self._apply_escalation(esc, job)
            elif esc.trigger == "step" and job is not None:
                if job.task.name == esc.task and job.pc == esc.trigger_arg:
                    self._apply_escalation(esc, job)

    def _live_job_of(self, task_name: str) -> Optional[Job]:
        live = [j for j in self.jobs
                if j.task.name == task_name and j.state != "done"]
        return max(live, key=lambda j: j.jid) if live else None

    def _apply_escalation(self, esc: Escalation, job: Job):
        esc.state = "active"
        esc.applied_at = self.now
        esc.target_jid = job.jid
        if esc.response is not None:
            esc.saved_deadline = job.abs_deadline
        self._recompute_escalation_effects()
        self._update_deferrals()

    def _end_escalations_of(self, job: Job):
        for esc in self.escalations:
            if esc.state == "active" and esc.target_jid == job.jid:
                esc.state = "done"
                esc.ended_at = job.completed_at
                esc.end_reason = "job completed"

    def _recompute_escalation_effects(self):
        for job in self.jobs:
            active = [e for e in self.escalations
                      if e.state == "active" and e.target_jid == job.jid]
            job.esc_boost = max((e.priority_raise for e in active), default=0)
            touched = any(
                e.target_jid == job.jid and e.saved_deadline is not None
                for e in self.escalations
            )
            if active or touched:
                tightened = [e.applied_at + e.response
                             for e in active if e.response is not None]
                job.abs_deadline = min([job.orig_deadline] + tightened)

    def _update_deferrals(self):
        active = [e for e in self.escalations
                  if e.state == "active" and e.strategy == "defer"]
        targets = {}
        for esc in active:
            t = next((j for j in self.jobs if j.jid == esc.target_jid), None)
            if t is not None and t.state != "done":
                targets[esc.eid] = t
        want = {}
        for job in self.jobs:
            if job.state not in ("ready", "running") or job.held:
                continue
            best = None
            for esc in active:
                t = targets.get(esc.eid)
                if t is None or t is job:
                    continue
                if job.eff_priority < t.eff_priority and (
                        best is None
                        or t.eff_priority > targets[best].eff_priority):
                    best = esc.eid
            if best is not None:
                want[job.jid] = best
        for job in self.jobs:
            desired = want.get(job.jid)
            if desired is not None and not job.deferred:
                job.deferred = True
                target = targets[desired]
                self.deferrals.append(DeferralRecord(
                    jid=job.jid,
                    task=job.task.name,
                    eid=desired,
                    deferred_at=self.now,
                    cause="升级 #%d 生效：任务 %s 关键程度临时提高（有效优先级 "
                          "%d），本作业有效优先级 %d 且未持有资源，按 defer "
                          "策略让步" % (desired, target.task.name,
                                        target.eff_priority,
                                        job.eff_priority),
                ))
            elif desired is None and job.deferred:
                job.deferred = False
                rec = next((r for r in reversed(self.deferrals)
                            if r.jid == job.jid and r.resumed_at is None),
                           None)
                if rec is not None:
                    rec.resumed_at = self.now
                    rec.reason = self._resume_reason(job, rec)

    def _resume_reason(self, job: Job, rec: DeferralRecord) -> str:
        if job.held:
            return "获得资源，须先完成临界区"
        esc = next((e for e in self.escalations if e.eid == rec.eid), None)
        if esc is not None and esc.state == "revoked":
            return "升级 #%d 已撤回" % esc.eid
        if esc is not None and esc.state == "done":
            return "升级 #%d 的目标作业已完成" % esc.eid
        return "不再低于任何生效升级的让步门槛"

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
        self._update_deferrals()

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
        self._check_escalation_triggers()
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
