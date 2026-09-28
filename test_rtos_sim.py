# 手算场景（时间单位任意）：
#   L: prio1, 释放0, 段[2, 3(持R), 1], 截止20
#   M: prio2, 释放4, 段[3],          截止20
#   H: prio3, 释放2, 段[2(需R)],     绝对截止7 <- 会错过
#   中断 tick: t=3, 耗时1
#
# 手算时间轴（基本优先级继承）：
#   0-2  L 段0
#   2    H 释放，需 R 被 L 持有 -> H 阻塞，L 继承 prio3
#   2-3  L 段1（继承优先级，M 在 t=4 释放也抢不过）
#   3-4  ISR tick
#   4-6  L 段1 剩余2，t=6 释放 R，H 解除阻塞，L 恢复 prio1
#   6-8  H 段0（持R），完成于8，错过截止7，超时1
#   8-11 M
#   11-12 L 段2，完成
import unittest
from rtos_sim import Simulator, Task, Section, Interrupt


def build():
    tasks = [
        Task("L", 1, 100, 20, [Section(2), Section(3, "R"), Section(1)]),
        Task("M", 2, 100, 20, [Section(3)], first_release=4),
        Task("H", 3, 100, 5,  [Section(2, "R")], first_release=2),  # 绝对截止 2+5=7
    ]
    return Simulator(tasks, [Interrupt("tick", 3, 1)], resources=["R"])


def merged(trace):
    out = []
    for s in trace:
        if out and out[-1][0] == s.who and abs(out[-1][2] - s.start) < 1e-9:
            out[-1] = (s.who, out[-1][1], float(s.end))
        else:
            out.append((s.who, float(s.start), float(s.end)))
    return out


class TestSim(unittest.TestCase):
    def test_hand_computed_trace(self):
        sim = build()
        sim.run(20)
        got = [(s.who, float(s.start), float(s.end)) for s in sim.trace]
        expect = [(w, float(a), float(b)) for w, a, b in
                  [("L", 0, 2), ("L", 2, 3), ("tick", 3, 4), ("L", 4, 6),
                   ("H", 6, 8), ("M", 8, 11), ("L", 11, 12)]]
        self.assertEqual(got, expect)

    def test_time_conservation(self):
        sim = build()
        sim.run(20)
        t = 0.0
        for s in sim.trace:
            self.assertAlmostEqual(s.start, t)   # 无缝隙、无重叠
            t = s.end
        self.assertAlmostEqual(t, 12.0)          # 所有工作完成后停止

    def test_priority_inheritance_blocks_medium(self):
        # 若无继承，M(prio2) 会在 t=4 抢占 L(prio1) 造成无界反转；
        # 有继承时 t=4..6 必须仍是 L 在跑。
        sim = build()
        sim.run(20)
        for s in sim.trace:
            if 4 < s.start < 6:
                self.assertEqual(s.who, "L")

    def test_miss_explanation(self):
        sim = build()
        sim.run(20)
        misses = sim.missed_deadlines()
        self.assertEqual(len(misses), 1)
        m = misses[0]
        self.assertEqual(m["task"], "H")
        self.assertAlmostEqual(m["missed_by"], 1.0)
        self.assertAlmostEqual(m["self_execution"], 2.0)
        self.assertAlmostEqual(m["blocked_by_lower_inversion"], 3.0)  # 2-3, 4-6
        self.assertAlmostEqual(m["isr_time"], 1.0)
        self.assertAlmostEqual(m["preempted_by_higher"], 0.0)
        print("\n--- 错过截止时间解释 ---\n" + m["explanation"])

    def test_snapshot_resume_matches_continuous(self):
        a = build(); a.run(20)
        b = build(); b.run(5)
        blob = b.snapshot()
        c = Simulator.restore(blob)
        c.run(20)
        self.assertEqual(merged(a.trace), merged(c.trace))
        # 恢复后错过分析也一致
        self.assertEqual([m["task"] for m in a.missed_deadlines()],
                         [m["task"] for m in c.missed_deadlines()])

    def test_extensible_record_for_future_power_model(self):
        # 任务/段记录带 attrs 扩展位：将来挂功耗参数不需要改结构
        t = Task("X", 1, 10, 10, [Section(1, attrs={"energy_nj": 5})],
                 attrs={"power_states": ["run", "sleep"]})
        self.assertEqual(t.sections[0].attrs["energy_nj"], 5)
        sim = Simulator([t])
        sim.run(5)
        blob = sim.snapshot()          # 扩展字段随快照一起序列化
        sim2 = Simulator.restore(blob)
        self.assertEqual(sim2.tasks["X"].attrs["power_states"], ["run", "sleep"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
