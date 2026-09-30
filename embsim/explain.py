"""错过截止时间的归因解释。

思路：把 [release, deadline] 窗口内的逐段执行记录分类汇总，
窗口里的每一拍都必须有归属——自己执行、系统开销（切换/系统调用）、
中断、被高优先级抢占、被低优先级阻塞（优先级反转）或空转。
各项之和恒等于窗口长度，因此“为什么错过”可以完全核算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .system import Job, System


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
    post_categories: Dict[str, int]   # 截止后到完成期间的占用分类（拍）
    post_interrupts: Dict[str, int]   # 截止后中断源名 -> 拍数
    accounted: int                    # 已归因总拍数
    window: int                       # 窗口长度 = deadline - release
    post_accounted: int = 0           # 截止后已归因拍数
    lines: List[str] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return self.accounted == self.window

    def report(self) -> str:
        return "\n".join(self.lines)


def explain_miss(sys: System, jid: int) -> MissExplanation:
    job = _find_job(sys, jid)
    release, deadline = job.release, job.abs_deadline
    window = deadline - release
    # 用截止瞬间冻结的剩余工作量，不能读 job.remaining_work()：
    # 作业可能在截止之后才完成，事后状态会把历史解释成“剩余零拍”。
    if job.deadline_remaining is not None:
        remaining = job.deadline_remaining
    else:
        rec = next((m for m in sys.misses if m.jid == jid), None)
        remaining = job.remaining_work() if rec is None else rec.remaining_work

    cats: Dict[str, int] = {
        "self_run": 0,        # 自己在执行计算
        "self_syscall": 0,    # 自己的 lock/unlock 系统调用
        "context_switch": 0,  # 上下文切换
        "interrupt": 0,       # 中断服务
        "idle": 0,            # 空转（自己未就绪/被阻塞且无人可跑）
        "preempted": 0,       # 被更高优先级任务抢占
        "inversion": 0,       # 低优先级任务运行造成的反转/阻塞
    }
    preemptions: Dict[str, int] = {}
    inversions: Dict[str, int] = {}
    interrupts: Dict[str, int] = {}
    post_categories: Dict[str, int] = {
        "self_run": 0, "self_syscall": 0, "context_switch": 0,
        "interrupt": 0, "idle": 0, "other": 0,
    }
    post_interrupts: Dict[str, int] = {}

    for seg in sys.trace:
        lo = max(seg.start, release)
        hi = min(seg.end, deadline)
        if hi > lo:
            n = hi - lo
            if seg.kind == "task" and seg.jid == jid:
                cats["self_run"] += n
            elif seg.kind in ("lock", "unlock") and seg.jid == jid:
                cats["self_syscall"] += n
            elif seg.kind == "switch":
                cats["context_switch"] += n
            elif seg.kind == "isr":
                cats["interrupt"] += n
                name = seg.detail.split(":", 1)[1]
                interrupts[name] = interrupts.get(name, 0) + n
            elif seg.kind == "idle":
                cats["idle"] += n
            elif seg.jid is not None:
                other = _find_job(sys, seg.jid)
                if other.base_priority >= job.base_priority:
                    cats["preempted"] += n
                    preemptions[other.task.name] = (
                        preemptions.get(other.task.name, 0) + n)
                else:
                    cats["inversion"] += n
                    inversions[other.task.name] = (
                        inversions.get(other.task.name, 0) + n)

        # 截止之后、作业完成之前的每一拍单独对账：这段时间不改变
        # “截止时还差几拍”，但解释了作业为什么到更晚才做完。
        if job.completed_at is not None:
            plo = max(seg.start, deadline)
            phi = min(seg.end, job.completed_at)
            if phi > plo:
                pn = phi - plo
                if seg.kind == "task" and seg.jid == jid:
                    post_categories["self_run"] += pn
                elif seg.kind in ("lock", "unlock") and seg.jid == jid:
                    post_categories["self_syscall"] += pn
                elif seg.kind == "switch":
                    post_categories["context_switch"] += pn
                elif seg.kind == "isr":
                    post_categories["interrupt"] += pn
                    name = seg.detail.split(":", 1)[1]
                    post_interrupts[name] = (
                        post_interrupts.get(name, 0) + pn)
                elif seg.kind == "idle":
                    post_categories["idle"] += pn
                else:
                    post_categories["other"] += pn

    accounted = sum(cats.values())
    post_accounted = sum(post_categories.values())
    lines = _build_lines(
        job, window, remaining, cats, preemptions, inversions,
        interrupts, accounted, post_categories, post_interrupts,
        post_accounted
    )
    return MissExplanation(
        jid=jid,
        task=job.task.name,
        release=release,
        deadline=deadline,
        remaining_work=remaining,
        categories=cats,
        preemptions=preemptions,
        inversions=inversions,
        interrupts=interrupts,
        post_categories=post_categories,
        post_interrupts=post_interrupts,
        accounted=accounted,
        window=window,
        post_accounted=post_accounted,
        lines=lines,
    )


def _find_job(sys: System, jid: int) -> Job:
    for job in sys.jobs:
        if job.jid == jid:
            return job
    raise KeyError("no job with jid=%r" % (jid,))


def _build_lines(job, window, remaining, cats, preemptions, inversions,
                 interrupts, accounted, post_categories, post_interrupts,
                 post_accounted) -> List[str]:
    lines = []
    lines.append(
        "任务 %s 的作业 #%d 错过截止时间：窗口 [%d, %d) 共 %d 拍，"
        "截止时还差 %d 拍计算。"
        % (job.task.name, job.jid, job.release, job.abs_deadline,
           window, remaining)
    )
    if job.completed_at is not None and job.completed_at > job.abs_deadline:
        detail = []
        if post_categories["self_run"]:
            detail.append("自身执行 %d 拍" % post_categories["self_run"])
        if post_categories["self_syscall"]:
            detail.append("系统调用 %d 拍" % post_categories["self_syscall"])
        if post_categories["context_switch"]:
            detail.append("切换 %d 拍" % post_categories["context_switch"])
        if post_categories["interrupt"]:
            names = ", ".join("%s %d拍" % kv
                              for kv in post_interrupts.items())
            detail.append("中断 %d 拍（%s）"
                          % (post_categories["interrupt"], names))
        if post_categories["other"]:
            detail.append("其他任务/ISR 占用 %d 拍"
                          % post_categories["other"])
        if post_categories["idle"]:
            detail.append("空转 %d 拍" % post_categories["idle"])
        lines.append(
            "- 该作业在截止之后才完成（%d 时刻），超期 %d 拍；"
            "完成事实不改变截止瞬间还差 %d 拍的历史记录。"
            "截止后逐段占用（%d 拍）：%s。"
            % (job.completed_at, job.completed_at - job.abs_deadline,
               remaining, post_accounted, "、".join(detail))
        )
    lines.append(
        "- 自身执行 %d 拍，系统调用 %d 拍"
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
    if cats["idle"]:
        lines.append("- 空转 %d 拍" % cats["idle"])
    lines.append(
        "合计 %d/%d 拍已归因%s。"
        % (accounted, window, "" if accounted == window else "（不一致！）")
    )
    return lines
