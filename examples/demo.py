# -*- coding: utf-8 -*-
"""演示：优先级反转 + 继承、中断开销、错过截止时间的归因、保存后继续。"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embsim import (Costs, InterruptSource, Mode, System, TaskSpec,
                    explain_miss)
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
print("=== 运行期切换工作模式：立即切换 vs 等当前作业结束 ===")
normal = Mode("normal", task_periods={"H": 10}, task_deadlines={"H": 10},
              lock_budgets={"R": 5})
boost = Mode("boost", task_periods={"H": 6}, task_deadlines={"H": 6},
             lock_budgets={"R": 1})


def mode_system():
    return System(
        tasks=[
            TaskSpec("H", 2, [lock("R"), run(2), unlock("R"), run(1)],
                     period=10),
            TaskSpec("L", 1, [run(8)], period=100),
        ],
        modes=[normal, boost], initial_mode="normal",
        costs=Costs(1, 1, 1),
    )


imm = mode_system()
imm.run_until(3)
imm.switch_mode("boost")          # 立即切换：旧作业承诺不变
imm.run_until(40)
wait = mode_system()
wait.run_until(3)
wait.switch_mode_after_jobs("boost")  # 等登记时已有的作业全部完成
wait.run_until(40)
c_imm = next(c for c in imm.mode_changes if c.new_mode == "boost")
c_wait = next(c for c in wait.mode_changes if c.new_mode == "boost")
print("立即切换生效时刻:", c_imm.time, "；等作业结束切换生效时刻:", c_wait.time)
old_imm = next(j for j in imm.jobs if j.jid == 0)
print("旧作业截止承诺保持:", old_imm.abs_deadline, "（释放模式",
      old_imm.release_mode, "）")
v = wait.budget_violations[-1]
print("锁预算越界（按拍所处模式记账）：作业#%d 的 %s 在模式「%s」持有 %d 拍 > 预算 %d"
      % (v.jid, v.resource, v.mode, v.held_in_mode, v.budget))
h_miss = next((m for m in imm.misses if m.task == "H"), None)
if h_miss is not None:
    print()
    print(explain_miss(imm, h_miss.jid).report())
