"""错过截止时间的归因解释。

思路：把 [release, deadline) 窗口内的逐段执行记录分类汇总，
窗口里的每一拍都必须有归属——自己执行、系统开销（切换/系统调用）、
中断、被高优先级抢占、被低优先级阻塞（优先级反转）或空转。
各项之和恒等于窗口长度，因此“为什么错过”可以完全核算。
注意：剩余工作量必须取自截止瞬间冻结的 MissRecord，不能事后读作业
当前状态，否则截止后补完的作业会被误解释成“剩余零拍”。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .system import Job, MissRecord, System


def _job_deferred_intervals(sys: System, job: Job):
    """由让步事件还原该作业处于延后态的 [t0, t1) 区间列表。"""
    intervals = []
    open_at = None
    for e in sys.concession_events:
        if e.jid != job.jid:
            continue
        if e.resumed_at is not None:
            intervals.append((e.time, e.resumed_at))
        else:
            end = sys.now
            state = next((j.state for j in sys.jobs
                          if j.jid == job.jid), None)
            if state == "deferred":
                intervals.append((e.time, end))
    return intervals


def _is_deferred_during(sys: System, job: Job, lo: int, hi: int) -> bool:
    for a, b in _job_deferred_intervals(sys, job):
        if lo >= a and hi <= b:
            return True
    return False


@dataclass
class MissExplanation:
    jid: int
    task: str
    release: int
    deadline: int
    remaining_work: int               # 截止时刻还差的 run 拍数
    categories: Dict[str, int]        # 类别 -> 拍数
    preemptions: Dict[str, int]       # 高优先级任务名 -> 拍数
    inversions: Dict[str, int]        # 低优先级任务名 -> 拍数（反转/阻塞）
    interrupts: Dict[str, int]        # 中断源名 -> 拍数
    mode_breakdown: Dict[str, Dict[str, int]]
    release_mode: str
    deadline_mode: str
    accounted: int                    # 已归因总拍数
    window: int                       # 窗口长度 = deadline - release
    wcet: int                         # 作业 WCET（run 拍总数）
    lines: List[str] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return self.accounted == self.window

    @property
    def work_consistent(self) -> bool:
        """自身已执行拍数 + 截止瞬间剩余拍数 必须恒等于 WCET。"""
        return self.categories["self_run"] + self.remaining_work == self.wcet

    def report(self) -> str:
        return "\n".join(self.lines)


def explain_miss(sys: System, jid: int) -> MissExplanation:
    job = _find_job(sys, jid)
    release, deadline = job.release, job.abs_deadline
    window = deadline - release
    miss = _find_miss(sys, jid)
    # 必须使用截止瞬间冻结的剩余工作量，不能事后再读 job.remaining_work()：
    # 作业可能在截止之后补完，事后状态会把“还差一拍”误报成“剩余零拍”。
    remaining_at_deadline = miss.remaining_work

    cats: Dict[str, int] = {
        "self_run": 0,        # 自己在执行计算
        "self_syscall": 0,    # 自己的 lock/unlock 系统调用
        "context_switch": 0,  # 上下文切换
        "interrupt": 0,       # 中断服务
        "idle": 0,            # 空转（自己未就绪/被阻塞且无人可跑）
        "preempted": 0,       # 被更高优先级任务抢占
        "inversion": 0,       # 低优先级任务运行造成的反转/阻塞
        "conceded": 0,        # 本作业因让步策略被推迟期间流逝的拍
    }
    preemptions: Dict[str, int] = {}
    inversions: Dict[str, int] = {}
    interrupts: Dict[str, int] = {}
    mode_breakdown: Dict[str, Dict[str, int]] = {}

    def add_mode_time(mode_name: str, category: str, amount: int):
        bucket = mode_breakdown.setdefault(mode_name, {})
        bucket[category] = bucket.get(category, 0) + amount

    for seg in sys.trace:
        lo = max(seg.start, release)
        hi = min(seg.end, deadline)
        if hi <= lo:
            continue
        n = hi - lo
        if seg.kind == "task" and seg.jid == jid:
            cats["self_run"] += n
            add_mode_time(seg.mode or "default", "self_run", n)
        elif seg.kind in ("lock", "unlock") and seg.jid == jid:
            cats["self_syscall"] += n
            add_mode_time(seg.mode or "default", "self_syscall", n)
        elif seg.kind == "switch":
            cats["context_switch"] += n
            add_mode_time(seg.mode or "default", "context_switch", n)
        elif seg.kind == "isr":
            cats["interrupt"] += n
            add_mode_time(seg.mode or "default", "interrupt", n)
            name = seg.detail.split(":", 1)[1]
            interrupts[name] = interrupts.get(name, 0) + n
        elif seg.kind == "idle":
            if _is_deferred_during(sys, job, lo, hi):
                cats["conceded"] += n
                add_mode_time(seg.mode or "default", "conceded", n)
            else:
                cats["idle"] += n
                add_mode_time(seg.mode or "default", "idle", n)
        elif seg.jid is not None:
            other = _find_job(sys, seg.jid)
            # 本作业正处于因关键程度升级而被延后的状态
            if _is_deferred_during(sys, job, lo, hi):
                cats["conceded"] += n
                add_mode_time(seg.mode or "default", "conceded", n)
            elif other.base_priority >= job.base_priority:
                cats["preempted"] += n
                add_mode_time(seg.mode or "default", "preempted", n)
                preemptions[other.task.name] = (
                    preemptions.get(other.task.name, 0) + n)
            else:
                cats["inversion"] += n
                add_mode_time(seg.mode or "default", "inversion", n)
                inversions[other.task.name] = (
                    inversions.get(other.task.name, 0) + n)

    accounted = sum(cats.values())
    lines = _build_lines(
        job, window, cats, preemptions, inversions, interrupts, accounted,
        remaining_at_deadline, mode_breakdown, miss.release_mode,
        miss.deadline_mode,
    )
    return MissExplanation(
        jid=jid,
        task=job.task.name,
        release=release,
        deadline=deadline,
        remaining_work=remaining_at_deadline,
        categories=cats,
        preemptions=preemptions,
        inversions=inversions,
        interrupts=interrupts,
        mode_breakdown=mode_breakdown,
        release_mode=miss.release_mode,
        deadline_mode=miss.deadline_mode,
        accounted=accounted,
        window=window,
        wcet=job.task.wcet,
        lines=lines,
    )


def _find_miss(sys: System, jid: int) -> MissRecord:
    for record in sys.misses:
        if record.jid == jid:
            return record
    raise KeyError("job with jid=%r has no deadline-miss record" % (jid,))


def _find_job(sys: System, jid: int) -> Job:
    for job in sys.jobs:
        if job.jid == jid:
            return job
    raise KeyError("no job with jid=%r" % (jid,))


def _build_lines(job, window, cats, preemptions, inversions,
                 interrupts, accounted, remaining, mode_breakdown,
                 release_mode, deadline_mode) -> List[str]:
    lines = []
    lines.append(
        "任务 %s 的作业 #%d 错过截止时间：窗口 [%d, %d) 共 %d 拍，"
        "截止瞬间还差 %d 拍计算（WCET %d 拍，窗口内已执行 %d 拍）。"
        % (job.task.name, job.jid, job.release, job.abs_deadline,
           window, remaining, job.task.wcet, cats["self_run"])
    )
    lines.append(
        "- 模式责任：作业释放在 %s，截止边界为 %s；超期量冻结在截止时刻"
        % (release_mode, deadline_mode)
    )
    lines.append(
        "- 自身执行 %d 拍（含系统调用 %d 拍）"
        % (cats["self_run"], cats["self_syscall"])
    )
    if cats["context_switch"]:
        lines.append("- 上下文切换开销 %d 拍" % cats["context_switch"])
    if cats["interrupt"]:
        detail = ", ".join("%s %d拍" % kv for kv in interrupts.items())
        lines.append("- 中断占用 %d 拍（%s）" % (cats["interrupt"], detail))
    if cats["preempted"]:
        detail = ", ".join("%s %d拍" % kv for kv in preemptions.items())
        lines.append("- 被高优先级抢占 %d 拍（%s）" % (cats["preempted"], detail))
    if cats["inversion"]:
        detail = ", ".join("%s %d拍" % kv for kv in inversions.items())
        lines.append(
            "- 优先级反转/阻塞 %d 拍（低优先级任务：%s）"
            % (cats["inversion"], detail)
        )
    if cats["conceded"]:
        lines.append(
            "- 因关键程度临时升级按让步策略被推迟 %d 拍"
            % cats["conceded"]
        )
    if cats["idle"]:
        lines.append("- 空转 %d 拍" % cats["idle"])
    if len(mode_breakdown) > 1 or any(
            mode_name != release_mode for mode_name in mode_breakdown):
        detail_parts = []
        for mode_name, bucket in mode_breakdown.items():
            detail_parts.append(
                "%s: %s" % (
                    mode_name,
                    ", ".join("%s=%d" % kv for kv in sorted(bucket.items())),
                )
            )
        lines.append("- 切换前后逐拍责任：" + "; ".join(detail_parts))
    work_ok = cats["self_run"] + remaining == job.task.wcet
    lines.append(
        "合计 %d/%d 拍已归因%s；工作量对账：窗口内自身执行 %d 拍 + "
        "截止瞬间剩余 %d 拍 = %d/%d 拍%s。"
        % (
            accounted, window,
            "" if accounted == window else "（不一致！）",
            cats["self_run"], remaining,
            cats["self_run"] + remaining, job.task.wcet,
            "" if work_ok else "（不一致！）",
        )
    )
    return lines
