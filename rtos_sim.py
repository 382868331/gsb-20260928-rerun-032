"""嵌入式任务仿真系统（第一版）"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Section:
    """作业内的一段连续执行。resource 非空表示执行本段须持有该资源。"""
    cost: float
    resource: Optional[str] = None
    attrs: dict = field(default_factory=dict)   # 扩展点：功耗模型挂这里


@dataclass
class Task:
    """周期任务，priority 数值越大优先级越高。"""
    name: str
    priority: int
    period: float
    deadline: float           # 相对截止时间
    sections: list[Section]
    first_release: float = 0.0
    attrs: dict = field(default_factory=dict)   # 扩展点：功耗模型挂这里

    @property
    def wcet(self) -> float:
        return sum(s.cost for s in self.sections)


@dataclass
class Interrupt:
    """在 time 时刻触发、占用 CPU cost 时间的 ISR。"""
    name: str
    time: float
    cost: float
    priority: int = 10_000    # ISR 之间的抢占优先级


@dataclass
class Resource:
    name: str
    holder: Optional[str] = None
    waiters: list = field(default_factory=list)   # FIFO 等待队列（任务名）


@dataclass
class Job:
    task: str
    seq: int
    release: float
    abs_deadline: float
    remaining: list           # 每段剩余执行时间
    section_idx: int = 0
    start: Optional[float] = None
    finish: Optional[float] = None
    blocked_on: Optional[str] = None


@dataclass
class Segment:
    """逐段执行记录：trace 的基本单元。"""
    start: float
    end: float
    who: str                  # 任务名 / ISR 名 / "IDLE"
    kind: str                 # "task" | "isr" | "idle"
    reason: str               # 本段为何结束


class Simulator:
    """单核事件驱动仿真器：抢占/阻塞/中断开销都计入时间轴。

    优先级反转采用基本优先级继承（可传递）：持有资源的任务被高优先级
    任务阻塞时，继承等待者中最高的有效优先级，释放资源后恢复。
    """

    def __init__(self, tasks, interrupts=None, resources=None,
                 context_switch_cost=0.0):
        self.tasks = {t.name: t for t in tasks}
        self.interrupts = sorted(interrupts or [], key=lambda i: (i.time, -i.priority))
        self.resources = {r: Resource(r) for r in (resources or [])}
        self.cs_cost = context_switch_cost
        self.now = 0.0
        self.trace = []
        self.jobs = []          # 已释放、未完成的作业
        self.done_jobs = []
        self._next_release = {t.name: t.first_release for t in tasks}
        self._seq = {t.name: 0 for t in tasks}
        self._isr_active = []   # 已触发未完成的 ISR: {name, remaining, priority}

    # ---------------- 优先级继承 ----------------
    def effective_priority(self, job):
        prio = self.tasks[job.task].priority
        for res in self.resources.values():
            if res.holder == job.task:
                for wname in res.waiters:
                    wj = self._find_unfinished(wname)
                    if wj is not None:
                        prio = max(prio, self.effective_priority(wj))
        return prio

    def _find_unfinished(self, task_name):
        for j in self.jobs:
            if j.task == task_name and j.finish is None:
                return j
        return None

    # ---------------- 主循环 ----------------
    def run(self, until):
        horizon = until
        while self.now < horizon - 1e-12:
            self._release_due()
            self._fire_interrupts()
            self._try_block()
            nxt = self._next_event_time(horizon)
            runner, kind = self._pick()
            if runner is None:
                if nxt is None or nxt >= horizon:
                    break
                self._emit(self.now, nxt, "IDLE", "idle", "无就绪任务")
                self.now = nxt
                continue
            if nxt is None:
                nxt = horizon
            budget = nxt - self.now
            if kind == "task":
                budget = min(budget, runner.remaining[runner.section_idx])
            else:
                budget = min(budget, runner["remaining"])
            start = self.now
            if kind == "task":
                job = runner
                if job.start is None:
                    job.start = start
                job.remaining[job.section_idx] -= budget
                self.now += budget
                if job.remaining[job.section_idx] <= 1e-12:
                    reason = self._advance_section(job)
                else:
                    reason = self._preempt_reason()
                self._emit(start, self.now, job.task, "task", reason)
            else:
                runner["remaining"] -= budget
                self.now += budget
                if runner["remaining"] <= 1e-12:
                    self._isr_active.remove(runner)
                    reason = "ISR %s 完成" % runner["name"]
                else:
                    reason = "被更高优先级 ISR 抢占"
                self._emit(start, self.now, runner["name"], "isr", reason)
        return self.trace

    def _next_event_time(self, horizon):
        cand = [t for t in self._next_release.values() if t > self.now + 1e-12]
        for isr in self.interrupts:
            if isr.time > self.now + 1e-12:
                cand.append(isr.time)
        cand = [t for t in cand if t <= horizon + 1e-12]
        return min(cand) if cand else None

    def _preempt_reason(self):
        for isr in self.interrupts:
            if abs(isr.time - self.now) < 1e-12:
                return "被中断 %s 抢占" % isr.name
        for name, t in self._next_release.items():
            if abs(t - self.now) < 1e-12:
                return "被新释放任务抢占（%s）" % name
        return "被抢占"

    def _advance_section(self, job):
        sec = self.tasks[job.task].sections[job.section_idx]
        if sec.resource:
            self._release_resource(sec.resource, job.task)
        job.section_idx += 1
        if job.section_idx >= len(job.remaining):
            job.finish = self.now
            self.jobs.remove(job)
            self.done_jobs.append(job)
            return "作业完成（截止 %.3f）" % job.abs_deadline
        return "段 %d 完成，进入段 %d" % (job.section_idx - 1, job.section_idx)

    def _try_block(self):
        """就绪作业若当前段需要资源：空闲则获取，否则进入阻塞队列。"""
        changed = True
        while changed:
            changed = False
            for job in list(self.jobs):
                if job.blocked_on or job.section_idx >= len(job.remaining):
                    continue
                sec = self.tasks[job.task].sections[job.section_idx]
                if sec.resource is None:
                    continue
                res = self.resources[sec.resource]
                if res.holder is None:
                    res.holder = job.task
                elif res.holder != job.task and job.task not in res.waiters:
                    res.waiters.append(job.task)
                    job.blocked_on = sec.resource
                    changed = True

    def _release_resource(self, rname, task):
        res = self.resources[rname]
        if res.holder == task:
            res.holder = None
            while res.waiters:
                nxt = res.waiters.pop(0)
                j = self._find_unfinished(nxt)
                if j is not None and j.blocked_on == rname:
                    j.blocked_on = None
                    res.holder = nxt
                    break

    def _pick(self):
        """ISR 优先于任务；任务按有效优先级，并列时先释放先服务。"""
        if self._isr_active:
            top = max(self._isr_active, key=lambda r: r["priority"])
            return top, "isr"
        ready = [j for j in self.jobs if j.blocked_on is None]
        if not ready:
            return None, None
        best = max(ready, key=lambda j: (self.effective_priority(j), -j.release))
        return best, "task"

    def _release_due(self):
        for name, task in self.tasks.items():
            while self._next_release[name] <= self.now + 1e-12:
                rel = self._next_release[name]
                job = Job(task=name, seq=self._seq[name], release=rel,
                          abs_deadline=rel + task.deadline,
                          remaining=[s.cost for s in task.sections])
                self.jobs.append(job)
                self._seq[name] += 1
                self._next_release[name] = rel + task.period

    def _fire_interrupts(self):
        for isr in self.interrupts:
            if abs(isr.time - self.now) < 1e-12 and not any(
                    r["name"] == isr.name for r in self._isr_active):
                self._isr_active.append(
                    {"name": isr.name, "remaining": isr.cost, "priority": isr.priority})

    def _emit(self, start, end, who, kind, reason):
        if end - start <= 1e-12:
            return
        self.trace.append(Segment(start, end, who, kind, reason))

    # ---------------- 截止时间分析 ----------------
    def missed_deadlines(self):
        out = []
        for job in self.done_jobs + self.jobs:
            if job.finish is not None and job.finish <= job.abs_deadline + 1e-12:
                continue
            if job.finish is None and self.now <= job.abs_deadline + 1e-12:
                continue
            out.append(self.explain_miss(job))
        return out

    def explain_miss(self, job):
        """把 [release, finish] 窗口拆成：自身执行/高优先级抢占/低优先级反转/中断。"""
        rel, dl = job.release, job.abs_deadline
        end = job.finish if job.finish is not None else max(self.now, dl)
        exec_t = isr_t = higher_t = block_t = 0.0
        myprio = self.tasks[job.task].priority
        for seg in self.trace:
            a, b = max(seg.start, rel), min(seg.end, end)
            if b <= a:
                continue
            if seg.who == job.task:
                exec_t += b - a
            elif seg.kind == "isr":
                isr_t += b - a
            elif seg.kind == "idle":
                pass
            else:
                hj = self._job_of(seg.who, a + 1e-9)
                if hj is not None and self.tasks[hj.task].priority < myprio:
                    block_t += b - a
                else:
                    higher_t += b - a
        miss_by = max(0.0, end - dl)
        return {
            "task": job.task, "seq": job.seq, "release": rel, "deadline": dl,
            "finish": job.finish, "missed_by": miss_by,
            "self_execution": exec_t, "preempted_by_higher": higher_t,
            "blocked_by_lower_inversion": block_t, "isr_time": isr_t,
            "explanation": (
                "任务 %s 第 %d 个作业在 %.3f 释放，截止 %.3f，%s，超时 %.3f。"
                "窗口内：自身执行 %.3f，被高优先级抢占 %.3f，"
                "低优先级持资源造成反转 %.3f，中断占用 %.3f。"
                % (job.task, job.seq, rel, dl,
                   ("完成于 %.3f" % job.finish) if job.finish is not None else "仍未完成",
                   miss_by, exec_t, higher_t, block_t, isr_t)),
        }

    def _job_of(self, task, t):
        for j in self.done_jobs + self.jobs:
            if j.task == task and j.release <= t and (j.finish is None or t < j.finish):
                return j
        return None

    # ---------------- 保存 / 恢复 ----------------
    def snapshot(self):
        def default(o):
            if hasattr(o, "__dict__"):
                return o.__dict__
            raise TypeError
        return json.dumps(self.__dict__, default=default).encode()

    @staticmethod
    def restore(blob):
        d = json.loads(blob)
        sim = Simulator.__new__(Simulator)
        sim.tasks = {k: Task(sections=[Section(**s) for s in v["sections"]],
                             **{kk: vv for kk, vv in v.items() if kk != "sections"})
                     for k, v in d["tasks"].items()}
        sim.interrupts = [Interrupt(**i) for i in d["interrupts"]]
        sim.resources = {k: Resource(**v) for k, v in d["resources"].items()}
        sim.cs_cost = d["cs_cost"]
        sim.now = d["now"]
        sim.trace = [Segment(**s) for s in d["trace"]]
        sim.jobs = [Job(**j) for j in d["jobs"]]
        sim.done_jobs = [Job(**j) for j in d["done_jobs"]]
        sim._next_release = d["_next_release"]
        sim._seq = {k: int(v) for k, v in d["_seq"].items()}
        sim._isr_active = d["_isr_active"]
        return sim
