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


class TestDeadlineInstantAccounting(unittest.TestCase):
    """A(wcet3, 相对截止2) 在 t=2 同刻被中断（服务2拍）抢占，t=6 才完成。

    手算：switch[0,1) A[1,2) isr[2,4) A[4,6) idle[6,8)
    截止时刻为 2，A 还差 2 拍；中断落在截止之后，不得计入窗口，
    但要在“截止后逐段占用”里逐拍对得上。
    """

    def build(self):
        return System(
            tasks=[TaskSpec("A", 1, [run(3)], period=100, deadline=2)],
            interrupts=[InterruptSource("T", [2], service=2)],
            costs=COSTS,
        )

    def test_frozen_remaining_and_post_deadline_breakdown(self):
        sys_ = self.build()
        sys_.run_until(8)
        job = sys_.jobs[0]
        self.assertEqual(job.completed_at, 6)
        self.assertEqual(len(sys_.misses), 1)
        miss = sys_.misses[0]
        # 截止瞬间冻结：还差 2 拍，事后完成不允许改写为 0
        self.assertEqual(miss.remaining_work, 2)
        self.assertEqual(job.deadline_remaining, 2)
        self.assertEqual(miss.completed_at, 6)

        ex = explain_miss(sys_, 0)
        self.assertEqual(ex.remaining_work, 2)
        self.assertTrue(ex.consistent)
        # 窗口 [0,2)：切换 1 + 自身执行 1；同刻中断不在窗口内
        self.assertEqual(ex.categories["self_run"], 1)
        self.assertEqual(ex.categories["context_switch"], 1)
        self.assertEqual(ex.categories["interrupt"], 0)
        # 截止后 [2,6) 共 4 拍：中断 2 + 自身执行 2
        self.assertEqual(ex.post_accounted, 4)
        self.assertEqual(ex.post_categories["interrupt"], 2)
        self.assertEqual(ex.post_categories["self_run"], 2)
        self.assertEqual(ex.post_interrupts, {"T": 2})
        report = ex.report()
        self.assertIn("截止时还差 2 拍", report)
        self.assertIn("截止之后才完成", report)

    def test_resume_cannot_change_miss_fact(self):
        """保存/重开后跑到作业完成，超期事实与冻结工作量保持不变。"""
        part = self.build()
        part.run_until(3)  # 错过已发生（t=2），作业尚未完成
        data = part.snapshot()
        resumed = System.restore(data)
        resumed.run_until(8)

        whole = self.build()
        whole.run_until(8)
        self.assertEqual(segs(whole), segs(resumed))
        self.assertEqual(len(resumed.misses), 1)
        self.assertEqual(resumed.misses[0].remaining_work, 2)
        self.assertEqual(resumed.misses[0].completed_at, 6)
        self.assertEqual(
            explain_miss(resumed, 0).remaining_work,
            explain_miss(whole, 0).remaining_work,
        )


class TestNestedLockInheritance(unittest.TestCase):
    """双锁嵌套等待链：H 等 R2（M 持有），M 等 R1（L 持有）。

    手算（含 K(优先级4) 在 t=11 释放抢占被提升的 L）：
      switch[0,1) L-lockR1[1,2) switch[2,3) M-lockR2[3,4) M[4,5)
      switch[5,6) H-lockR2[6,7)(阻塞) switch[7,8) M-lockR1[8,9)(阻塞)
      switch[9,10) L[10,11) switch[11,12) K[12,14) switch[14,15)
      L[15,16) L-unlockR1[16,17) switch[17,18) M-lockR1[18,19) M[19,20)
      M-unlockR1[20,21) M-unlockR2[21,22) switch[22,23) H-lockR2[23,24)
      H[24,25) H-unlockR2[25,26)
    """

    def build(self):
        return System(tasks=nested_chain_tasks(with_k=True), costs=COSTS)

    def test_transitive_propagation_preemption_and_restore(self):
        sys_ = self.build()
        sys_.run_until(12)
        l, m, h, k = sys_.jobs
        # 传递继承：H(3)->M 提升到 3(boost1)，M->L 提升到 4(boost2)，
        # 因此 t=11 时即便 K(4) 释放，也能立刻抢占正在借优先级跑的 L。
        self.assertEqual((h.boost, h.state, h.blocked_on), (0, "blocked", "R2"))
        self.assertEqual((m.boost, m.state, m.blocked_on), (1, "blocked", "R1"))
        self.assertEqual((l.boost, l.state), (2, "ready"))
        self.assertEqual((k.state,), ("running",))

        sys_.run_until(26)
        self.assertEqual(
            segs(sys_),
            [
                (0, 1, "switch", "switch->L#0"),
                (1, 2, "lock", "L#0 lock R1"),
                (2, 3, "switch", "switch->M#1"),
                (3, 4, "lock", "M#1 lock R2"),
                (4, 5, "task", "M#1"),
                (5, 6, "switch", "switch->H#2"),
                (6, 7, "lock", "H#2 lock R2"),
                (7, 8, "switch", "switch->M#1"),
                (8, 9, "lock", "M#1 lock R1"),
                (9, 10, "switch", "switch->L#0"),
                (10, 11, "task", "L#0"),
                (11, 12, "switch", "switch->K#3"),
                (12, 14, "task", "K#3"),
                (14, 15, "switch", "switch->L#0"),
                (15, 16, "task", "L#0"),
                (16, 17, "unlock", "L#0 unlock R1"),
                (17, 18, "switch", "switch->M#1"),
                (18, 19, "lock", "M#1 lock R1"),
                (19, 20, "task", "M#1"),
                (20, 21, "unlock", "M#1 unlock R1"),
                (21, 22, "unlock", "M#1 unlock R2"),
                (22, 23, "switch", "switch->H#2"),
                (23, 24, "lock", "H#2 lock R2"),
                (24, 25, "task", "H#2"),
                (25, 26, "unlock", "H#2 unlock R2"),
            ],
        )
        done = {j.task.name: j.completed_at for j in sys_.jobs}
        self.assertEqual(done, {"L": 17, "M": 22, "H": 26, "K": 14})
        # 解锁后沿等待关系逐层恢复：所有提升归零
        self.assertTrue(all(j.boost == 0 for j in sys_.jobs))

    def test_no_transitive_inheritance_when_disabled(self):
        """关闭继承：H 阻塞在 R2 上不会沿等待链提升 M、L。"""
        sys_ = System(tasks=nested_chain_tasks(with_k=False),
                      costs=COSTS, use_inheritance=False)
        sys_.run_until(11)
        self.assertTrue(all(j.boost == 0 for j in sys_.jobs))


def nested_chain_tasks(with_k):
    tasks = [
        TaskSpec("L", 1,
                 [lock("R1"), run(2), unlock("R1")],
                 period=100),
        TaskSpec("M", 2,
                 [lock("R2"), run(1), lock("R1"), run(1),
                  unlock("R1"), unlock("R2")],
                 period=100, offset=2),
        TaskSpec("H", 3,
                 [lock("R2"), run(1), unlock("R2")],
                 period=100, offset=5),
    ]
    if with_k:
        tasks.append(TaskSpec("K", 4, [run(2)], period=100, offset=11))
    return tasks


if __name__ == "__main__":
    unittest.main()
