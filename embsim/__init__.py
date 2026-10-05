"""嵌入式任务仿真系统（第一版）。

虚拟处理器 + 任务/中断模型 + 可抢占固定优先级调度，
共享资源采用立即优先级继承（immediate priority inheritance）。
"""

from .model import (
    CriticalityCommitment,
    Costs,
    InterruptSource,
    ModeSpec,
    Step,
    TaskSpec,
    lock,
    run,
    unlock,
)
from .system import (
    CommitmentEvent,
    ConcessionEvent,
    Job,
    ManualInterventionRecord,
    MissRecord,
    Segment,
    System,
)
from .explain import MissExplanation, explain_miss

__all__ = [
    "Costs",
    "CriticalityCommitment",
    "InterruptSource",
    "ModeSpec",
    "Step",
    "TaskSpec",
    "lock",
    "run",
    "unlock",
    "Job",
    "CommitmentEvent",
    "ConcessionEvent",
    "ManualInterventionRecord",
    "MissRecord",
    "Segment",
    "System",
    "MissExplanation",
    "explain_miss",
]
