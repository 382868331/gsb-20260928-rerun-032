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
    priority: int = 0     # ISR 自身优先级；高优先级可嵌套抢占低优先级 ISR

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
class ModeSpec:
    """运行期工作模式。

    periods/deadlines 只影响切换后新释放的作业；已经释放的作业继续使用
    释放时承诺的周期和截止时间。lock_budgets 按资源名限制持锁时长，
    未列出的资源表示不限。
    """

    name: str
    periods: Dict[str, int] = field(default_factory=dict)
    deadlines: Dict[str, int] = field(default_factory=dict)
    lock_budgets: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------- 关键程度承诺

@dataclass
class CriticalityCommitment:
    """运行期临时关键程度承诺（让步策略的规格）。

    name:       承诺标识；同一承诺可激活、撤回，多次提出按先后排队。
    task:       获得临时高关键程度的任务名。
    priority:   承诺生效期间该任务作业的有效关键级别（数值越大越高）。
    deadline:   更紧的相对响应要求；只约束生效期间新释放的作业。
    strategy:   对其他低关键作业的让步策略：
                "inherit_only" 仅靠优先级继承，不主动延后任何作业；
                "defer"       主动延后更低关键的就绪作业，但处于
                              锁等待链上的持有者豁免（必须先把锁放出）。
    """

    name: str
    task: str
    priority: Optional[int] = None
    deadline: Optional[int] = None
    strategy: str = "defer"

    def __post_init__(self):
        if self.strategy not in ("inherit_only", "defer"):
            raise ValueError("unknown concession strategy: %r" % self.strategy)
