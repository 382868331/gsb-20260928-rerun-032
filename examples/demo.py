# -*- coding: utf-8 -*-
"""演示：优先级反转 + 继承、中断开销、错过截止时间的归因、保存后继续。"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embsim import (Costs, CriticalityCommitment, InterruptSource, System,
                    TaskSpec, explain_miss)
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
print("=== 多把锁的嵌套等待链：继承沿等待关系传播 ===")
chain = System(
    tasks=[
        TaskSpec("L", 1, [lock("Q"), run(5), unlock("Q")], period=100),
        TaskSpec("M", 2, [lock("R"), run(1), lock("Q"), run(1),
                          unlock("Q"), unlock("R")],
                 period=100, offset=2),
        TaskSpec("H", 3, [lock("R"), run(1), unlock("R")],
                 period=100, offset=5),
    ],
    costs=Costs(1, 1, 1),
)
chain.run_until(9)
j = {x.task.name: x for x in chain.jobs}
print("M 被 H 等 R 顶起：eff =", j["M"].eff_priority, "boost =", j["M"].boost)
print("L 沿 M->Q 等待链被顶起：eff =", j["L"].eff_priority,
      "boost =", j["L"].boost)
chain.run_until(25)
j = {x.task.name: x for x in chain.jobs}
print("解除后恢复：M boost =", j["M"].boost, "，H 完成于",
      j["H"].completed_at)

print()
print("=== 临时升级 H 的关键程度：低关键任务让步，持锁部分继续放锁 ===")
esc = System(
    tasks=[
        TaskSpec("L", 1,
                 [lock("R"), run(3), unlock("R"), run(2)], period=100),
        TaskSpec("M", 2, [run(4)], period=100, offset=1, deadline=7),
        TaskSpec("H", 3,
                 [run(1), lock("R"), run(1), unlock("R")],
                 period=100, offset=4),
    ],
    costs=Costs(context_switch=0, lock=1, unlock=1),
)
esc.escalate(CriticalityCommitment("hot", "H", priority=5,
                                   deadline=8, strategy="defer"),
             at_time=4)
esc.withdraw("hot", at_time=12)
esc.run_until(14)
for c in esc.concession_timeline():
    print("  %s# 推迟@%d -> 恢复@%s：%s（%s）"
          % (c.task, c.time, c.resumed_at, c.reason, c.resume_reason))
m_job = next(j for j in esc.jobs if j.task.name == "M")
if m_job.missed:
    print()
    print(explain_miss(esc, m_job.jid).report())

print()
print("=== 两种让步策略对比 ===")
base = System(
    tasks=[
        TaskSpec("L", 1, [run(4)], period=100),
        TaskSpec("H", 3, [run(2)], period=100, offset=2),
    ],
    costs=Costs(context_switch=0),
)
base.run_until(2)
cmp = base.compare_concessions(
    CriticalityCommitment("hot", "H", priority=5), horizon=8
)
for name, r in cmp.items():
    print("  %-12s 让步项=%d 完成时刻=%s 超期=%d"
          % (name, len(r["concessions"]), r["completed"],
             len(r["misses"])))
