# -*- coding: utf-8 -*-
"""可手算核对的小例子测试。

每个用例的时间线都在注释里逐拍推出，与仿真结果比对。
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embsim import (
    Costs,
    InterruptSource,
    System,
    TaskSpec,
    explain_miss,
    lock,
    run,
    unlock,
)

COSTS = Costs(context_switch=1, lock=1, unlock=1)


def segs(sys_):
    return [(s.start, s.end, s.kind, s.detail) for s in sys_.trace]


def assert_contiguous(test, sys_, until):
    tr = sys_.trace
    test.assertEqual(tr[0].start, 0)
    for a, b in zip(tr, tr[1:]):
        test.assertEqual(a.end, b.start)
    test.assertEqual(tr[-1].end, until)


class TestPreemption(unittest.TestCase):
    """H(优先级2,周期10,wcet2) 与 L(优先级1,周期100,wcet4)，切换开销1。

    手算：switch[0,1) H[1,3) switch[3,4) L[4,8) idle[8,10)
          switch[10,11) H[11,13) idle[13,14)
    """

    def test_timeline(self):
        sys_ = System(
            tasks=[
                TaskSpec("H", 2, [run(2)], period=10),
                TaskSpec("L", 1, [run(4)], period=100),
            ],
            costs=COSTS,
        )
        sys_.run_until(14)
        self.assertEqual(
            segs(sys_),
            [
                (0, 1, "switch", "switch->H#0"),
                (1, 3, "task", "H#0"),
                (3, 4, "switch", "switch->L#1"),
                (4, 8, "task", "L#1"),
                (8, 10, "idle", "idle"),
                (10, 11, "switch", "switch->H#2"),
                (11, 13, "task", "H#2"),
                (13, 14, "idle", "idle"),
            ],
        )
        done = {j.jid: j.completed_at for j in sys_.jobs}
        self.assertEqual(done[0], 3)   # H 第 1 拍末开始，第 3 拍末完成
        self.assertEqual(done[1], 8)
        self.assertEqual(done[2], 13)
        self.assertEqual(sys_.misses, [])
        assert_contiguous(self, sys_, 14)


def inversion_tasks(h_deadline=None):
    return [
        TaskSpec("L", 1, [lock("R"), run(3), unlock("R"), run(1)], period=100),
        TaskSpec("M", 2, [run(3)], period=100, offset=2),
        TaskSpec("H", 3, [run(1), lock("R"), run(1), unlock("R")],
                 period=100, offset=3, deadline=h_deadline),
    ]


class TestPriorityInheritance(unittest.TestCase):
    """L 持锁被 H 阻塞；继承开启时 L 提升到 H 的优先级，赶在 M 前放锁。

    手算（继承开启）：
      switch[0,1) L-lock[1,2) switch[2,3) switch[3,4) H[4,5) H-lock[5,6)(阻塞)
      switch[6,7) L[7,10) L-unlock[10,11) switch[11,12)
      H-lock[12,13) H[13,14) H-unlock[14,15) -> H 完成于 15
      switch[15,16) M[16,19) switch[19,20) L[20,21)
    """

    def test_inheritance_on(self):
        sys_ = System(tasks=inversion_tasks(), costs=COSTS,
                      use_inheritance=True)
        sys_.run_until(21)
        done = {j.task.name: j.completed_at for j in sys_.jobs}
        self.assertEqual(done["H"], 15)
        self.assertEqual(done["M"], 19)
        self.assertEqual(done["L"], 21)
        # L 在 [7,10) 区间是“借”了 H 的优先级在跑
        self.assertIn((7, 10, "task", "L#0"), segs(sys_))
        assert_contiguous(self, sys_, 21)

    def test_inheritance_off(self):
        """不继承：M 抢在 L 前运行，H 被多阻塞 4 拍，完成于 19。"""
        sys_ = System(tasks=inversion_tasks(), costs=COSTS,
                      use_inheritance=False)
        sys_.run_until(21)
        done = {j.task.name: j.completed_at for j in sys_.jobs}
        self.assertEqual(done["M"], 10)
        self.assertEqual(done["H"], 19)
        self.assertEqual(done["L"], 21)


class TestInterrupt(unittest.TestCase):
    """A(wcet5) + t=2 触发的中断（进入1+服务2+退出1）。

    手算：switch[0,1) A[1,2) isr[2,6) A[6,10)
    """

    def test_isr_timeline(self):
        sys_ = System(
            tasks=[TaskSpec("A", 1, [run(5)], period=100)],
            interrupts=[InterruptSource("Timer", [2], service=2,
                                        entry=1, exit=1)],
            costs=COSTS,
        )
        sys_.run_until(11)
        self.assertEqual(
            segs(sys_),
            [
                (0, 1, "switch", "switch->A#0"),
                (1, 2, "task", "A#0"),
                (2, 6, "isr", "isr:Timer"),
                (6, 10, "task", "A#0"),
                (10, 11, "idle", "idle"),
            ],
        )
        self.assertEqual(sys_.jobs[0].completed_at, 10)


class TestDeadlineMissExplanation(unittest.TestCase):
    """继承场景下 H 的相对截止设为 10（绝对 13）：完成于 15，错过，且窗口可完全归因。"""

    def test_explain(self):
        sys_ = System(tasks=inversion_tasks(h_deadline=10), costs=COSTS,
                      use_inheritance=True)
        sys_.run_until(21)
        self.assertEqual(len(sys_.misses), 1)
        miss = sys_.misses[0]
        self.assertEqual(miss.task, "H")
        self.assertEqual(miss.remaining_work, 1)

        ex = explain_miss(sys_, miss.jid)
        self.assertTrue(ex.consistent)
        self.assertEqual(ex.window, 10)
        # 窗口 [3,13)：自身运行1 + 自身系统调用2 + 切换3 + 反转4
        self.assertEqual(ex.categories["self_run"], 1)
        self.assertEqual(ex.categories["self_syscall"], 2)
        self.assertEqual(ex.categories["context_switch"], 3)
        self.assertEqual(ex.categories["inversion"], 4)
        self.assertEqual(ex.inversions, {"L": 4})
        self.assertIn("优先级反转", ex.report())


class TestSnapshotResume(unittest.TestCase):
    """保存状态后继续运行，结果必须与一口气跑完一致。"""

    def scenario(self):
        return System(
            tasks=inversion_tasks(),
            interrupts=[InterruptSource("Tick", [5, 12], service=1)],
            costs=COSTS,
        )

    def test_resume_equivalent(self):
        whole = self.scenario()
        whole.run_until(30)

        part = self.scenario()
        part.run_until(9)
        data = part.snapshot()
        part2 = System.restore(data)
        part2.run_until(30)

        self.assertEqual(segs(whole), segs(part2))
        self.assertEqual(
            [(m.jid, m.task, m.deadline) for m in whole.misses],
            [(m.jid, m.task, m.deadline) for m in part2.misses],
        )
        done_whole = {j.jid: j.completed_at for j in whole.jobs}
        done_part = {j.jid: j.completed_at for j in part2.jobs}
        self.assertEqual(done_whole, done_part)

    def test_save_load_file(self):
        part = self.scenario()
        part.run_until(7)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.pkl")
            part.save(path)
            loaded = System.load(path)
        loaded.run_until(30)

        whole = self.scenario()
        whole.run_until(30)
        self.assertEqual(segs(whole), segs(loaded))


if __name__ == "__main__":
    unittest.main()
