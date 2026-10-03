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

from .model import (
    Costs,
    CriticalityCommitment,
    InterruptSource,
    ModeSpec,
    TaskSpec,
)


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
    state: str = "ready"             # ready/running/blocked/deferred/done
    held: List[str] = field(default_factory=list)
    blocked_on: Optional[str] = None
    lock_request_mode: Optional[str] = None
    boost: int = 0                   # 继承得到的优先级增量
    deferred_by: Optional[str] = None  # 被哪条承诺延后（承诺名）
    deferral_sources: List[str] = field(default_factory=list)
    release_commitment: Optional[str] = None  # 释放时生效的升级承诺
    completed_at: Optional[int] = None
    missed: bool = False
    blocked_ticks: int = 0
    delayed_ticks: int = 0
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
    release_commitment: Optional[str] = None
    deadline_commitments: tuple = ()


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
class CommitmentEvent:
    """升级/撤回在调度轴上的生效事件（保留先后关系）。"""

    time: int
    seq: int
    kind: str                # escalate / withdraw
    name: str                # 承诺名


@dataclass
class ConcessionEvent:
    """某项作业因让步策略被推迟、以及后来恢复的记录。"""

    time: int                # 开始被延后的时刻（拍边界）
    jid: int
    task: str
    commitment: str          # 哪条承诺导致延后
    commitment_task: str     # 被升级的是哪个任务
    strategy: str
    reason: str
    resumed_at: Optional[int] = None
    resume_reason: Optional[str] = None


@dataclass
class WaitEdge:
    """一条“等待者持有请求 -> 当前资源持有者”的活动关系。"""

    generation: int
    waiter_jid: int
    waiter_task: str
    holder_jid: int
    holder_task: str
    resource: str
    created_at: int
    request_mode: str
    acquired_mode: str
    active: bool = True
    lost_at: Optional[int] = None
    lost_reason: Optional[str] = None


@dataclass
class TemporaryExecutionGrant:
    """沿真正阻止资源释放的等待链传递的一次临时执行机会。"""

    time: int
    generation: int
    source_kind: str       # priority / commitment
    commitment: Optional[str]
    root_waiter_jid: int
    root_waiter_task: str
    holder_jid: int
    holder_task: str
    resource: str
    level: int
    path: tuple
    in_cycle: bool = False
    active: bool = True
    lost_at: Optional[int] = None
    lost_reason: Optional[str] = None


@dataclass
class WaitCycle:
    time: int
    generation: int
    cycle: tuple
    resources: tuple
    active: bool = True
    resolved_at: Optional[int] = None
    resolution: Optional[str] = None


class System:
    def __init__(
        self,
        tasks: List[TaskSpec],
        interrupts: Optional[List[InterruptSource]] = None,
        costs: Costs = Costs(),
        use_inheritance: bool = True,
        use_transfer: Optional[bool] = None,
        modes: Optional[List[ModeSpec]] = None,
        initial_mode: Optional[str] = None,
        commitments: Optional[List[CriticalityCommitment]] = None,
    ):
        self.tasks = {t.name: t for t in tasks}
        self.interrupts = interrupts or []
        self.costs = costs
        self.use_inheritance = use_inheritance
        self.use_transfer = use_inheritance if use_transfer is None else use_transfer
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

        # 关键程度承诺：规格表 + 当前生效表 + 调度轴上的生效事件
        self.commitments: Dict[str, CriticalityCommitment] = {}
        if commitments:
            for c in commitments:
                if c.name in self.commitments:
                    raise ValueError("duplicate commitment name: %r" % c.name)
                if c.task not in self.tasks:
                    raise KeyError("unknown task in commitment: %r" % c.task)
                self.commitments[c.name] = c
        self.active_commitments: Dict[str, CriticalityCommitment] = {}
        self.pending_commitment_events: List[CommitmentEvent] = []
        self.commitment_events: List[CommitmentEvent] = []
        self.concession_events: List[ConcessionEvent] = []
        self._next_commitment_seq = 0
        self.wait_edges: List[WaitEdge] = []
        self.execution_grants: List[TemporaryExecutionGrant] = []
        self.wait_cycles: List[WaitCycle] = []
        self._next_wait_generation = 0

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

    # ------------------------------------------------------------ 承诺升级/撤回

    def escalate(
        self,
        commitment: CriticalityCommitment,
        at_time: Optional[int] = None,
    ):
        """提出一项关键程度升级；可在 at_time 指定的拍边界生效。

        已经登记过同名承诺时沿用原规格（再次提出只重排生效顺序）；
        撤回与升级按提出先后排队，同一承诺可反复升级、撤回。
        请求本身不占 CPU，不回滚任何已占有资源或已消耗执行时间。
        """
        if commitment.task not in self.tasks:
            raise KeyError("unknown task in commitment: %r" % commitment.task)
        if at_time is not None and at_time < self.now:
            raise ValueError("escalation time cannot be in the past")
        self.commitments.setdefault(commitment.name, commitment)
        self._queue_commitment_event(
            "escalate", commitment.name,
            self.now if at_time is None else at_time,
        )

    def withdraw(self, name: str, at_time: Optional[int] = None):
        """撤回一项承诺；撤回到点后被它延后的作业按顺序恢复。"""
        if name not in self.commitments:
            raise KeyError("unknown commitment: %r" % name)
        if at_time is not None and at_time < self.now:
            raise ValueError("withdrawal time cannot be in the past")
        self._queue_commitment_event(
            "withdraw", name,
            self.now if at_time is None else at_time,
        )

    def _queue_commitment_event(self, kind: str, name: str, time: int):
        event = CommitmentEvent(
            time=time,
            seq=self._next_commitment_seq,
            kind=kind,
            name=name,
        )
        self._next_commitment_seq += 1
        self.pending_commitment_events.append(event)
        self.pending_commitment_events.sort(key=lambda e: (e.time, e.seq))

    def _apply_due_commitment_events(self):
        due = [e for e in self.pending_commitment_events if e.time <= self.now]
        if not due:
            return
        remaining = [e for e in self.pending_commitment_events
                     if e.time > self.now]
        due.sort(key=lambda e: (e.time, e.seq))
        self.pending_commitment_events = remaining
        for event in due:
            spec = self.commitments[event.name]
            if event.kind == "escalate":
                self.active_commitments[event.name] = spec
            else:
                self.active_commitments.pop(event.name, None)
            self.commitment_events.append(event)
        # 升级/撤回改变等待者的关键级别与每条等待链的临时执行权
        self._sync_wait_graph()
        # 承诺集合变化后重新评估谁应当被让步
        self._recompute_deferrals()

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
                # 生效中的升级承诺可以收紧新作业的相对响应要求；
                # 已释放作业的绝对截止承诺冻结在 Job 里，事后不改。
                release_commitment = None
                for c in self.active_commitments.values():
                    if c.task == t.name and c.deadline is not None:
                        rel_deadline = min(rel_deadline, c.deadline)
                        release_commitment = c.name
                self._next_release[t.name] += period
                job = Job(
                    jid=self._next_jid,
                    task=t,
                    release=self.now,
                    abs_deadline=self.now + rel_deadline,
                    period=period,
                    rel_deadline=rel_deadline,
                    mode=release_mode,
                    release_commitment=release_commitment,
                )
                if t.program and t.program[0].kind == "run":
                    job.step_left = t.program[0].arg
                self._next_jid += 1
                self.jobs.append(job)
                self._recompute_deferrals()
        while self._irq_schedule and self._irq_schedule[0][0] <= self.now:
            _, _, src = self._irq_schedule.pop(0)
            self._irq_pending.append({"src": src, "left": src.total_cost})

    # ------------------------------------------------------------ 优先级继承

    def _job_static_level(self, job: Optional[Job]) -> int:
        if job is None:
            return -10**18
        level = job.base_priority
        for c in self._job_commitments(job):
            if c.priority is not None:
                level = max(level, c.priority)
        return level

    def _active_wait_map(self) -> Dict[int, List[WaitEdge]]:
        return {
            j.jid: [
                e for e in self.wait_edges
                if e.active and e.waiter_jid == j.jid
            ]
            for j in self.jobs if j.state != "done"
        }

    def _sync_wait_graph(self):
        """按当前 holder/waiters 增删等待边，并同步逐项临时执行授权。"""
        expected = set()
        for res_name, res in self.resources.items():
            holder = res["holder"]
            for waiter in res["waiters"]:
                if holder is None or waiter is holder:
                    continue
                expected.add((waiter.jid, holder.jid, res_name))

        current = {
            (e.waiter_jid, e.holder_jid, e.resource): e
            for e in self.wait_edges if e.active
        }
        for key, edge in current.items():
            if key not in expected:
                edge.active = False
                edge.lost_at = self.now
                edge.lost_reason = "等待关系消失"
        for waiter_jid, holder_jid, res_name in sorted(expected):
            key = (waiter_jid, holder_jid, res_name)
            if key in current:
                continue
            waiter = self.jobs[waiter_jid]
            holder = self.jobs[holder_jid]
            res = self.resources[res_name]
            edge = WaitEdge(
                generation=self._next_wait_generation,
                waiter_jid=waiter_jid,
                waiter_task=waiter.task.name,
                holder_jid=holder_jid,
                holder_task=holder.task.name,
                resource=res_name,
                created_at=self.now,
                request_mode=waiter.lock_request_mode or self.mode,
                acquired_mode=res.get("mode") or self.mode,
            )
            self._next_wait_generation += 1
            self.wait_edges.append(edge)
        self._refresh_execution_grants()
        self._refresh_wait_cycles()

    def _outgoing_edges(self) -> Dict[int, List[WaitEdge]]:
        result: Dict[int, List[WaitEdge]] = {}
        for edge in self.wait_edges:
            if edge.active:
                result.setdefault(edge.waiter_jid, []).append(edge)
        return result

    def _detect_cycles(self) -> List[tuple]:
        outgoing = self._outgoing_edges()
        cycles = []
        seen_cycles = set()
        for start in sorted(outgoing):
            stack = [(start, [start], [])]
            while stack:
                node, nodes, edges = stack.pop()
                for edge in outgoing.get(node, []):
                    nxt = edge.holder_jid
                    if nxt == start and nodes:
                        canonical_nodes = tuple(nodes)
                        canonical_edges = tuple(
                            sorted((e.resource, e.waiter_jid, e.holder_jid)
                                   for e in edges + [edge])
                        )
                        if canonical_edges not in seen_cycles:
                            seen_cycles.add(canonical_edges)
                            cycles.append((canonical_nodes,
                                           tuple(e.resource for e in edges + [edge])))
                    elif nxt not in nodes:
                        stack.append((nxt, nodes + [nxt], edges + [edge]))
        return cycles

    def _refresh_wait_cycles(self):
        found = self._detect_cycles()
        found_keys = {tuple(sorted(zip(nodes, resources)))
                      for nodes, resources in found}
        for record in self.wait_cycles:
            if not record.active:
                continue
            key = tuple(sorted(zip(record.cycle, record.resources)))
            if key not in found_keys:
                record.active = False
                record.resolved_at = self.now
                record.resolution = "等待环对应边消失"
        known = {
            tuple(sorted(zip(r.cycle, r.resources))): r
            for r in self.wait_cycles
        }
        for nodes, resources in found:
            key = tuple(sorted(zip(nodes, resources)))
            if key in known:
                continue
            self.wait_cycles.append(WaitCycle(
                time=self.now,
                generation=self._next_wait_generation,
                cycle=nodes,
                resources=resources,
            ))
            self._next_wait_generation += 1

    def _root_sources(self, job: Job) -> List[tuple]:
        sources = []
        if self.use_inheritance and job.blocked_on is not None:
            sources.append(("priority", None, job.base_priority, job))
        if self.use_transfer:
            for commitment in self._job_commitments(job):
                if commitment.priority is not None and job.blocked_on is not None:
                    sources.append(("commitment", commitment.name,
                                    commitment.priority, job))
        return sources

    def _refresh_execution_grants(self):
        outgoing = self._outgoing_edges()
        cycle_nodes = {
            jid for cycle in self.wait_cycles if cycle.active
            for jid in cycle.cycle
        }
        wanted = []
        for root in self.jobs:
            if root.state == "done" or root.blocked_on is None:
                continue
            for kind, commitment_name, level, source_job in self._root_sources(root):
                stack = [(root, (root.jid,), ())]
                while stack:
                    waiter, node_path, edge_path = stack.pop()
                    for edge in outgoing.get(waiter.jid, []):
                        if edge.holder_jid in node_path:
                            continue
                        path = edge_path + (edge.generation,)
                        wanted.append((kind, commitment_name, root.jid,
                                       edge.holder_jid, edge.resource, level,
                                       path, edge.holder_jid in cycle_nodes))
                        holder = self.jobs[edge.holder_jid]
                        if holder.blocked_on is not None:
                            stack.append((holder, node_path + (holder.jid,),
                                          path))

        active = {
            (g.source_kind, g.commitment, g.root_waiter_jid, g.holder_jid,
             g.resource, g.path): g
            for g in self.execution_grants if g.active
        }
        wanted_keys = set()
        for kind, commitment_name, root_jid, holder_jid, resource, level, path, in_cycle in wanted:
            key = (kind, commitment_name, root_jid, holder_jid, resource, path)
            wanted_keys.add(key)
            existing = active.get(key)
            if existing is None:
                self.execution_grants.append(TemporaryExecutionGrant(
                    time=self.now,
                    generation=self._next_wait_generation,
                    source_kind=kind,
                    commitment=commitment_name,
                    root_waiter_jid=root_jid,
                    root_waiter_task=self.jobs[root_jid].task.name,
                    holder_jid=holder_jid,
                    holder_task=self.jobs[holder_jid].task.name,
                    resource=resource,
                    level=level,
                    path=path,
                    in_cycle=in_cycle,
                ))
                self._next_wait_generation += 1
            else:
                existing.in_cycle = in_cycle
                existing.level = level
        for key, grant in active.items():
            if key not in wanted_keys:
                grant.active = False
                grant.lost_at = self.now
                grant.lost_reason = "等待边、升级请求或资源归属已失效"

        if not self.use_transfer and not self.use_inheritance:
            for job in self.jobs:
                job.boost = 0
            return
        for job in self.jobs:
            want = 0
            for grant in self.execution_grants:
                if grant.active and grant.holder_jid == job.jid \
                        and not grant.in_cycle:
                    want = max(want, grant.level - job.base_priority)
            job.boost = max(0, want)

    # ------------------------------------------------------------ 关键程度与让步

    def _job_commitments(self, job: Job) -> List[CriticalityCommitment]:
        """当前对该作业生效的升级承诺（同一任务可能被多条承诺升级）。"""
        return [c for c in self.active_commitments.values()
                if c.task == job.task.name and job.state != "done"]

    def _job_crit_level(self, job: Optional[Job]) -> int:
        """作业的关键级别：静态优先级之上叠加有效承诺。

        阻塞在锁上的已升级作业同样沿等待链施压（见 _protected_holders）。
        """
        if job is None:
            return -10**18
        level = job.eff_priority
        for c in self._job_commitments(job):
            if c.priority is not None:
                level = max(level, c.priority)
        return level

    def _protected_holders_for(self, commitment: Optional[CriticalityCommitment]) -> set:
        roots = [
            j for j in self.jobs
            if j.state not in ("done", "deferred")
            and j.blocked_on is not None
            and (commitment is None or j.task.name == commitment.task)
        ]
        protected = set()
        stack = roots
        seen = {j.jid for j in roots}
        outgoing = self._outgoing_edges()
        while stack:
            waiter = stack.pop()
            for edge in outgoing.get(waiter.jid, []):
                holder = self.jobs[edge.holder_jid]
                protected.add(holder.jid)
                if holder.blocked_on is not None and holder.jid not in seen:
                    seen.add(holder.jid)
                    stack.append(holder)
        return protected

    def _recompute_deferrals(self):
        """依据当前生效承诺重新决定每个作业的延后/恢复。

        - inherit_only 策略从不主动延后任何作业；
        - defer 策略把关键级别严格低于最高已升级级别、且不持锁处于
          等待链上的就绪/运行作业延后；持锁链上的持有者豁免，先把锁放出；
        - 承诺撤回后不再满足延后条件的作业立即恢复就绪。
        延后不清空 pc/step_left/held：已占有资源与已执行时间原样保留。
        """
        deferring = [c for c in self.active_commitments.values()
                     if c.strategy == "defer"]
        if not deferring:
            for job in self.jobs:
                for name in list(job.deferral_sources):
                    self._remove_deferral_source(job, name, "承诺撤回，让步结束")
            return

        top_level = max(
            (c.priority for c in deferring if c.priority is not None),
            default=None,
        )
        protected_by_any = set()
        for commitment in deferring:
            protected_by_any.update(self._protected_holders_for(commitment))
        for job in self.jobs:
            if job.state in ("done", "blocked"):
                continue
            valid_sources = set()
            protected_sources = set()
            for commitment in deferring:
                target = next(
                    (j for j in self.jobs
                     if j.task.name == commitment.task
                     and j.state not in ("done", "deferred")),
                    None,
                )
                candidate = self._deferring_commitment_for(
                    job, [commitment], top_level)
                protected = self._protected_holders_for(commitment)
                job_has_commitment = any(
                    other.task == job.task.name for other in deferring)
                if job_has_commitment:
                    candidate = None
                elif target is None or target.state == "done":
                    candidate = None
                elif job.held and job.jid not in protected:
                    candidate = None
                if candidate is not None and job.jid not in protected:
                    valid_sources.add(commitment.name)
                elif candidate is not None:
                    protected_sources.add(commitment.name)
            for name in valid_sources:
                if name not in job.deferral_sources:
                    self._add_deferral_source(job, self.commitments[name])
            for name in list(job.deferral_sources):
                if name in valid_sources:
                    continue
                existing_spec = self.commitments.get(name)
                protected_by_existing = (
                    job.jid in self._protected_holders_for(existing_spec)
                    if existing_spec is not None else False
                )
                if protected_by_existing:
                    reason = "作业持有升级任务正在等待的锁，先完成临界区放锁"
                    protected_sources.add(name)
                else:
                    reason = "承诺撤回、资源释放或工作模式切换后该项让步失去依据"
                self._remove_deferral_source(job, name, reason)

    def _deferring_commitment_for(self, job, deferring, top_level):
        """返回要求该作业让步的承诺（取提出最早者），无则 None。"""
        candidates = []
        job_level = self._job_crit_level(job)
        for c in deferring:
            # 被升级任务自身、以及同级或更高关键作业不被延后
            if c.task == job.task.name:
                continue
            if any(other.task == job.task.name for other in deferring
                   if other is not c):
                continue
            if c.priority is not None and job_level >= c.priority:
                continue
            if top_level is not None and job_level >= top_level:
                continue
            candidates.append(c)
        if not candidates:
            return None
        candidates.sort(
            key=lambda c: next(
                (e.seq for e in self.commitment_events[::-1]
                 if e.kind == "escalate" and e.name == c.name),
                10**18,
            )
        )
        return candidates[0]

    def _add_deferral_source(self, job: Job, commitment: CriticalityCommitment):
        if commitment.name in job.deferral_sources:
            return
        if self.current is job and job.state == "running":
            job.state = "ready"
            self.current = None
        reason = (
            "关键级别 %d 低于升级任务 %s 的承诺级别 %d，按 %s 策略延后；"
            "已执行进度与已占有资源保留"
            % (
                job.base_priority,
                commitment.task,
                commitment.priority,
                commitment.strategy,
            )
        )
        job.deferral_sources.append(commitment.name)
        job.state = "deferred"
        job.deferred_by = job.deferral_sources[0]
        # 若正在切换到该作业的路上，取消目标（已付的切换开销留在时间线上）
        if self._switch_to is job:
            self._switch_to = None
            self._switch_left = 0
        self.concession_events.append(ConcessionEvent(
            time=self.now,
            jid=job.jid,
            task=job.task.name,
            commitment=commitment.name,
            commitment_task=commitment.task,
            strategy=commitment.strategy,
            reason=reason,
        ))

    def _remove_deferral_source(self, job: Job, commitment_name: str,
                                reason: str):
        if commitment_name not in job.deferral_sources:
            return
        job.deferral_sources.remove(commitment_name)
        event = next(
            (e for e in reversed(self.concession_events)
             if e.jid == job.jid and e.commitment == commitment_name
             and e.resumed_at is None),
            None,
        )
        if event is not None:
            event.resumed_at = self.now
            event.resume_reason = reason
        if not job.deferral_sources:
            job.state = "ready"
            job.deferred_by = None
        else:
            job.deferred_by = job.deferral_sources[0]

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
        self._sync_wait_graph()
        self._recompute_deferrals()

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
                if w.state == "deferred":
                    pass  # 仍处于让步延后：不参与本次抢锁
                else:
                    w.state = "ready"
                w.blocked_on = None
            res["waiters"].clear()
        job.pc += 1
        self._prime_step(job)
        self._sync_wait_graph()
        # 放锁后等待链改变：原本被豁免的持有者可能变成可延后作业，
        # 原本延后的等待者若仍被承诺覆盖则继续延后（状态在重算中统一裁决）
        self._recompute_deferrals()

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
        # deferred 作业的资源与执行进度仍保留，但不参与调度
        return [j for j in self.jobs
                if j.state in ("ready", "running")]

    def _select(self) -> Optional[Job]:
        ready = self._ready_jobs()
        if not ready:
            return None
        # 选择键先看关键级别（静态优先级 + 继承 + 生效承诺），
        # 同级别按作业号（FIFO，保证调度轴确定）
        return max(ready, key=lambda j: (self._job_crit_level(j), -j.jid))

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
        self._sync_wait_graph()
        self._recompute_deferrals()

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
        self._apply_due_commitment_events()
        self._release_due()
        for job in self.jobs:
            if job.state == "blocked":
                job.blocked_ticks += 1
            elif job.state == "deferred":
                job.delayed_ticks += 1

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
        # 拍边界：预排到此刻的升级/撤回在同一边界生效，
        # 与模式切换的先后按各自请求顺序独立保留（互不改写历史）
        self._apply_due_commitment_events()
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
                    release_commitment=job.release_commitment,
                    deadline_commitments=tuple(sorted(self.active_commitments)),
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

    # ------------------------------------------------------------ 让步查询

    def concession_timeline(self) -> List[ConcessionEvent]:
        """逐项返回谁被推迟、何时恢复及原因（按生效先后）。"""
        return list(self.concession_events)

    def pending_deferrals(self) -> List[Dict]:
        """截至当前时刻仍未恢复的被延后作业，附承诺与原因说明。"""
        result = []
        for job in self.jobs:
            if job.state != "deferred":
                continue
            event = next(
                (e for e in reversed(self.concession_events)
                 if e.jid == job.jid and e.resumed_at is None),
                None,
            )
            result.append({
                "jid": job.jid,
                "task": job.task.name,
                "deferred_at": event.time if event else None,
                "commitment": job.deferred_by,
                "commitment_sources": list(job.deferral_sources),
                "commitment_task": (
                    self.commitments[job.deferred_by].task
                    if job.deferred_by in self.commitments else None
                ),
                "strategy": (
                    self.commitments[job.deferred_by].strategy
                    if job.deferred_by in self.commitments else None
                ),
                "reason": event.reason if event else None,
                "remaining_work": job.remaining_work(),
                "held_resources": list(job.held),
                "consumed_delay_ticks": job.delayed_ticks,
            })
        return result

    def resource_ownership(self) -> List[Dict]:
        """返回当前资源归属、等待者和获得资源时冻结的模式预算。"""
        result = []
        for name in sorted(self.resources):
            res = self.resources[name]
            holder = res["holder"]
            result.append({
                "resource": name,
                "holder": holder.task.name if holder else None,
                "holder_jid": holder.jid if holder else None,
                "waiters": [
                    {"task": w.task.name, "jid": w.jid,
                     "request_mode": w.lock_request_mode}
                    for w in res["waiters"]
                ],
                "acquired_at": res["acquired_at"],
                "acquired_mode": res["mode"],
                "budget": res["budget"],
            })
        return result

    def wait_graph(self, active_only: bool = True) -> List[WaitEdge]:
        """返回等待边；历史边保留其消失时刻和原因。"""
        return [e for e in self.wait_edges if e.active or not active_only]

    def execution_grant_timeline(self, active_only: bool = True) -> List[TemporaryExecutionGrant]:
        """逐项返回沿等待链产生的临时执行权及其来源。"""
        return [g for g in self.execution_grants
                if g.active or not active_only]

    def active_wait_cycles(self) -> List[WaitCycle]:
        return [c for c in self.wait_cycles if c.active]

    def wait_cycle_reports(self) -> List[Dict]:
        """说明哪些环仍真实卡住；提级不能替代锁释放。"""
        reports = []
        for cycle in self.wait_cycles:
            if not cycle.active:
                continue
            reports.append({
                "time": cycle.time,
                "jobs": tuple(self.jobs[j].task.name for j in cycle.cycle
                              if 0 <= j < len(self.jobs)),
                "resources": cycle.resources,
                "resolvable_by_priority": False,
                "reason": "环上每个作业都在等待下一作业已占有的资源，"
                          "提高优先级不会释放任何锁；需要撤回请求、"
                          "打破资源顺序或让仿真明确停留在阻塞态。",
            })
        return reports

    def job_scheduling_summary(self) -> Dict[str, Dict]:
        return {
            job.task.name: {
                "jid": job.jid,
                "state": job.state,
                "blocked_ticks": job.blocked_ticks,
                "delayed_ticks": job.delayed_ticks,
                "remaining_work": job.remaining_work(),
                "held_resources": list(job.held),
                "completed_at": job.completed_at,
                "missed": job.missed,
            }
            for job in self.jobs
        }

    def compare_transfer(self, horizon: int) -> Dict[str, Dict]:
        """比较关闭/启用临时执行权传递的调度结果。"""
        results = {}
        for enabled in (False, True):
            clone = System.restore(self.snapshot())
            clone.use_inheritance = enabled
            clone.use_transfer = enabled
            clone.run_until(horizon)
            key = "transfer" if enabled else "original"
            results[key] = {
                "completed": {
                    j.task.name: j.completed_at
                    for j in clone.jobs if j.completed_at is not None
                },
                "jobs": clone.job_scheduling_summary(),
                "ownership": clone.resource_ownership(),
                "cycles": clone.wait_cycle_reports(),
                "grants": clone.execution_grant_timeline(False),
                "system": clone,
            }
        return results

    def compare_concessions(
        self,
        commitment: CriticalityCommitment,
        horizon: int,
    ) -> Dict[str, Dict]:
        """同一升级分别用两种让步策略各跑一份副本，便于用户比较。

        返回 {"inherit_only": {...}, "defer": {...}}，每项汇总完成时刻、
        超期情况与推迟/恢复明细；副本不影响当前系统。
        """
        results = {}
        for strategy in ("inherit_only", "defer"):
            clone = System.restore(self.snapshot())
            spec = CriticalityCommitment(
                name=commitment.name,
                task=commitment.task,
                priority=commitment.priority,
                deadline=commitment.deadline,
                strategy=strategy,
            )
            clone.commitments.pop(commitment.name, None)
            clone.escalate(spec)
            clone.run_until(horizon)
            results[strategy] = {
                "misses": [
                    {
                        "jid": m.jid, "task": m.task,
                        "deadline": m.deadline,
                        "remaining_work": m.remaining_work,
                    }
                    for m in clone.misses
                ],
                "completed": {
                    j.task.name: j.completed_at
                    for j in clone.jobs if j.completed_at is not None
                },
                "concessions": [
                    {
                        "time": e.time, "task": e.task,
                        "resumed_at": e.resumed_at,
                        "reason": e.reason,
                        "resume_reason": e.resume_reason,
                    }
                    for e in clone.concession_events
                ],
                "still_deferred": clone.pending_deferrals(),
                "system": clone,
            }
        return results

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
