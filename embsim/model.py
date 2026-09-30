"""任务、资源与中断的静态模型。

设计要点（为后续功耗模型预留）：
- TaskSpec 只描述“任务是什么”，不掺杂调度/功耗策略；
  attrs 是开放的扩展字典，将来功耗参数（动态功耗、静态功耗、
  不同频率下的能耗系数等）直接挂在 attrs 上即可，
  任务记录结构本身不需要重新设计。
- 任务体用 program（指令步序列）表达，资源临界区是程序的一部分，
  因此阻塞、解锁等开销天然落在时间线上，而不是事后附加。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------- 程序步

@dataclass(frozen=True)
class Step:
    """任务程序中的一步。kind ∈ {run, lock, unlock}。"""

    kind: str
    arg: object  # run -> 整数拍数；lock/unlock -> 资源名


def run(ticks: int) -> Step:
    assert ticks > 0
    return Step("run", ticks)


def lock(resource: str) -> Step:
    return Step("lock", resource)


def unlock(resource: str) -> Step:
    return Step("unlock", resource)


# ---------------------------------------------------------------- 任务

@dataclass
class TaskSpec:
    """周期性任务描述（任务记录）。

    priority: 静态优先级，数值越大优先级越高。
    program:  指令步序列；WCET = 所有 run 步之和。
    period / deadline / offset: 周期、相对截止时间、首次释放偏移。
    attrs:    开放扩展字段（功耗参数等将来直接放这里）。
    """

    name: str
    priority: int
    program: List[Step]
    period: int
    deadline: Optional[int] = None  # 缺省等于周期
    offset: int = 0
    attrs: Dict[str, Any] = field(default_factory=dict)

    @property
    def wcet(self) -> int:
        return sum(s.arg for s in self.program if s.kind == "run")

    @property
    def rel_deadline(self) -> int:
        return self.period if self.deadline is None else self.deadline


# ---------------------------------------------------------------- 中断

@dataclass
class InterruptSource:
    """中断源：在 times 指定的时刻触发，服务程序耗时不计入任何任务。"""

    name: str
    times: List[int]
    service: int          # ISR 主体耗时（拍）
    entry: int = 0        # 进入开销（保存现场等）
    exit: int = 0         # 退出开销（恢复现场等）
    priority: int = 0     # 中断优先级（大者高）：高者可嵌套抢占低者

    @property
    def total_cost(self) -> int:
        return self.entry + self.service + self.exit


# ---------------------------------------------------------------- 开销参数

@dataclass(frozen=True)
class Costs:
    """虚拟处理器的机制开销（全部以拍计，占用 CPU 时间）。

    context_switch: 任务间切换一次的开销；
    lock / unlock:    资源系统调用的开销（无论成功与否都要付）。
    """

    context_switch: int = 1
    lock: int = 1
    unlock: int = 1


# ---------------------------------------------------------------- 资源

@dataclass
class Resource:
    """互斥资源。持有者/等待队列在 System 运行态中维护。"""

    name: str


# ---------------------------------------------------------------- 工作模式

@dataclass
class Mode:
    """运行期工作模式。

    模式只决定“切换之后新释放作业”的调度参数与锁占用预算，
    绝不回改切换之前已经释放的作业（其周期承诺/截止时刻保持不变）。

    task_periods:   各任务在该模式下的周期；缺省回落到 TaskSpec.period。
    task_deadlines: 各任务在该模式下的相对截止；缺省回落到 TaskSpec.deadline。
    lock_budgets:   每把锁在该模式下允许的连续持有预算（拍）。
                    持锁跨过模式边界后，占用拍按“当前模式”记账，
                    因此跨模式持锁可能触发新模式的预算告警。
    """

    name: str
    task_periods: Dict[str, int] = field(default_factory=dict)
    task_deadlines: Dict[str, Optional[int]] = field(default_factory=dict)
    lock_budgets: Dict[str, int] = field(default_factory=dict)
    attrs: Dict[str, Any] = field(default_factory=dict)

    def period_of(self, task: TaskSpec) -> int:
        return self.task_periods.get(task.name, task.period)

    def deadline_of(self, task: TaskSpec) -> int:
        rel = self.task_deadlines.get(task.name, task.deadline)
        return task.period if rel is None else rel

    def budget_of(self, resource: str) -> Optional[int]:
        return self.lock_budgets.get(resource)
