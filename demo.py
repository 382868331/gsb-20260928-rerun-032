"""演示：优先级继承 + 中断 + 错过截止时间解释 + 快照续跑。"""
from rtos_sim import Simulator, Task, Section, Interrupt

tasks = [
    Task("L", 1, 100, 20, [Section(2), Section(3, "R"), Section(1)]),
    Task("M", 2, 100, 20, [Section(3)], first_release=4),
    Task("H", 3, 100, 5,  [Section(2, "R")], first_release=2),
]
sim = Simulator(tasks, [Interrupt("tick", 3, 1)], resources=["R"])

sim.run(5)                       # 先跑到 t=5
blob = sim.snapshot()            # 保存状态
sim = Simulator.restore(blob)    # 恢复
sim.run(20)                      # 继续运行

print("=== 逐段执行记录 ===")
for s in sim.trace:
    print(f"[{s.start:5.2f}, {s.end:5.2f})  {s.who:5s}  {s.reason}")

print("\n=== 错过截止时间分析 ===")
for m in sim.missed_deadlines():
    print(m["explanation"])
