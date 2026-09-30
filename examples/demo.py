# -*- coding: utf-8 -*-
"""演示：优先级反转 + 继承、中断开销、错过截止时间的归因、保存后继续。"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embsim import Costs, InterruptSource, System, TaskSpec, explain_miss
from embsim import lock, run, unlock


def build(h_deadline):
    return System(
        tasks=[
            TaskSpec("L", 1, [lock("R"), run(3), unlock("R"), run(1)],
                     period=100),
            TaskSpec("M", 2, [run(3)], period=100, offset=2),
            TaskSpec("H", 3, [run(1), lock("R"), run(1), unlock("R")],
                     period=100, offset=3, deadline=h_deadline),
        ],
        interrupts=[InterruptSource("Tick", [9], service=1)],
        costs=Costs(context_switch=1, lock=1, unlock=1),
        use_inheritance=True,
    )


def show(sys_):
    for s in sys_.trace:
        print("  [%2d, %2d)  %-7s %s" % (s.start, s.end, s.kind, s.detail))


print("=== 场景：H 相对截止 10（继承开启，含一次中断） ===")
sys_ = build(h_deadline=10)
sys_.run_until(25)
show(sys_)
for miss in sys_.misses:
    print()
    print(explain_miss(sys_, miss.jid).report())

print()
print("=== 保存状态后继续运行 ===")
sys2 = build(h_deadline=None)
sys2.run_until(8)
snap = sys2.snapshot()
sys2b = System.restore(snap)
sys2b.run_until(25)
whole = build(h_deadline=None)
whole.run_until(25)
same = [ (s.start,s.end,s.kind,s.detail) for s in sys2b.trace ] == \
       [ (s.start,s.end,s.kind,s.detail) for s in whole.trace ]
print("断点续跑与一次跑完的执行记录一致:", same)


print()
print("=== 多把锁嵌套等待：H 等 R2(M)、M 等 R1(L)，K 同刻抢占被提升的 L ===")
chain = System(
    tasks=[
        TaskSpec("L", 1, [lock("R1"), run(2), unlock("R1")], period=100),
        TaskSpec("M", 2, [lock("R2"), run(1), lock("R1"), run(1),
                          unlock("R1"), unlock("R2")], period=100, offset=2),
        TaskSpec("H", 3, [lock("R2"), run(1), unlock("R2")],
                 period=100, offset=5),
        TaskSpec("K", 4, [run(2)], period=100, offset=11),
    ],
    costs=Costs(context_switch=1, lock=1, unlock=1),
)
chain.run_until(26)
for s in chain.trace:
    print("  [%2d, %2d)  %-7s %s" % (s.start, s.end, s.kind, s.detail))
print("完成时刻:",
      {j.task.name: j.completed_at for j in chain.jobs},
"；全部 boost 已恢复:", all(j.boost == 0 for j in chain.jobs))
