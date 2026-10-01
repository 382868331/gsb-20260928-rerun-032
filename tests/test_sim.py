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
    ModeSpec,
    System,
    TaskSpec,
    explain_deferrals,
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


def nested_chain_tasks():
    """多把锁的嵌套等待链：L 持 Q；M 持 R 后等 Q；H 等 R。

    继承沿等待关系传播：H(3) 等 R -> M 提升到 3；
    M(已为 3) 又等 Q -> L 也提升到 3。
    """
    return [
        TaskSpec("L", 1, [lock("Q"), run(5), unlock("Q")], period=100),
        TaskSpec("M", 2, [lock("R"), run(1), lock("Q"), run(1),
                          unlock("Q"), unlock("R")],
                 period=100, offset=2),
        TaskSpec("H", 3, [lock("R"), run(1), unlock("R")],
                 period=100, offset=5),
    ]


class TestNestedChainInheritance(unittest.TestCase):
    """逐拍（开销全 1）：

    switch[0,1) L-lockQ[1,2) switch[2,3) M-lockR[3,4) M[4,5)
    switch[5,6) H-lockR[6,7)(阻塞；M 立即继承到 3)
    switch[7,8) M-lockQ[8,9)(阻塞；继承沿链传播，L 提升到 3)
    switch[9,10) L[10,15) L-unlockQ[15,16)(L 完成，解除继承)
    switch[16,17) M-lockQ[17,18) M[18,19) M-unlockQ[19,20)
    M-unlockR[20,21)(M 完成) switch[21,22) H-lockR[22,23)
    H[23,24) H-unlockR[24,25)
    """

    def _jobs(self, sys_):
        return {j.task.name: j for j in sys_.jobs}

    def test_timeline_and_boost_propagation(self):
        sys_ = System(tasks=nested_chain_tasks(), costs=COSTS)
        sys_.run_until(9)  # 推进到 M 阻塞在 Q 的拍末
        jobs = self._jobs(sys_)
        # 直接等待：H 等 R（M 持有），M 提升到 H 的优先级
        self.assertEqual(jobs["H"].state, "blocked")
        self.assertEqual(jobs["M"].boost, 1)
        self.assertEqual(jobs["M"].eff_priority, 3)
        # 传递等待：M(有效3) 等 Q（L 持有），L 沿链提升到 3
        self.assertEqual(jobs["M"].blocked_on, "Q")
        self.assertEqual(jobs["L"].boost, 2)
        self.assertEqual(jobs["L"].eff_priority, 3)

        sys_.run_until(15)  # L 以提升后的优先级跑完 5 拍临界区
        jobs = self._jobs(sys_)
        self.assertEqual(jobs["L"].boost, 2)
        self.assertEqual(jobs["L"].eff_priority, 3)

        sys_.run_until(16)  # L 放 Q 并完成：解除继承，boost 归零
        jobs = self._jobs(sys_)
        self.assertEqual(jobs["L"].completed_at, 16)
        self.assertEqual(jobs["L"].boost, 0)
        # M 仍因 H 等 R 而保持继承来的优先级 3
        self.assertEqual(jobs["M"].eff_priority, 3)
        self.assertEqual(jobs["M"].boost, 1)

        sys_.run_until(25)
        jobs = self._jobs(sys_)
        # M 放掉 R 后恢复自身优先级；H 得以跑完
        self.assertEqual(jobs["M"].completed_at, 21)
        self.assertEqual(jobs["H"].completed_at, 25)
        self.assertEqual(jobs["M"].boost, 0)
        self.assertEqual(jobs["H"].boost, 0)
        self.assertEqual(
            segs(sys_),
            [
                (0, 1, "switch", "switch->L#0"),
                (1, 2, "lock", "L#0 lock Q"),
                (2, 3, "switch", "switch->M#1"),
                (3, 4, "lock", "M#1 lock R"),
                (4, 5, "task", "M#1"),
                (5, 6, "switch", "switch->H#2"),
                (6, 7, "lock", "H#2 lock R"),
                (7, 8, "switch", "switch->M#1"),
                (8, 9, "lock", "M#1 lock Q"),
                (9, 10, "switch", "switch->L#0"),
                (10, 15, "task", "L#0"),
                (15, 16, "unlock", "L#0 unlock Q"),
                (16, 17, "switch", "switch->M#1"),
                (17, 18, "lock", "M#1 lock Q"),
                (18, 19, "task", "M#1"),
                (19, 20, "unlock", "M#1 unlock Q"),
                (20, 21, "unlock", "M#1 unlock R"),
                (21, 22, "switch", "switch->H#2"),
                (22, 23, "lock", "H#2 lock R"),
                (23, 24, "task", "H#2"),
                (24, 25, "unlock", "H#2 unlock R"),
            ],
        )
        assert_contiguous(self, sys_, 25)

    def test_no_inheritance_no_chain_boost(self):
        """关闭继承：不存在任何 boost 传播；L 放锁前 H 一直阻塞在 R 上。"""
        sys_ = System(tasks=nested_chain_tasks(), costs=COSTS,
                      use_inheritance=False)
        sys_.run_until(16)
        jobs = self._jobs(sys_)
        self.assertEqual(jobs["L"].boost, 0)
        self.assertEqual(jobs["M"].boost, 0)
        self.assertEqual(jobs["H"].state, "blocked")
        self.assertEqual(jobs["H"].blocked_on, "R")

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

    def test_report_frozen_at_deadline(self):
        """截止瞬间还差一拍；作业在截止之后补完，报告不得改口为零拍。"""
        sys_ = System(tasks=inversion_tasks(h_deadline=10), costs=COSTS,
                      use_inheritance=True)
        sys_.run_until(21)  # H 完成于 15（截止 13 之后）
        h_job = next(j for j in sys_.jobs if j.task.name == "H")
        self.assertEqual(h_job.completed_at, 15)
        self.assertEqual(h_job.remaining_work(), 0)  # 事后状态：已经做完

        ex = explain_miss(sys_, h_job.jid)
        # 错过记录与文字解释都以截止瞬间冻结的剩余量为准
        self.assertEqual(sys_.misses[0].remaining_work, 1)
        self.assertEqual(ex.remaining_work, 1)
        self.assertIn("截止瞬间还差 1 拍", ex.report())
        self.assertTrue(ex.work_consistent)
        # 错过记录、时间轴、文字解释三者工作量对账：1 已执行 + 1 剩余 = WCET 2
        self.assertEqual(
            ex.categories["self_run"] + ex.remaining_work, h_job.task.wcet
        )

    def test_report_stable_after_snapshot_reopen(self):
        """保存重开后继续生成报告，不能改变已经发生的超期事实。"""
        sys_ = System(tasks=inversion_tasks(h_deadline=10), costs=COSTS)
        sys_.run_until(21)
        frozen = [(m.jid, m.task, m.deadline, m.remaining_work)
                  for m in sys_.misses]
        data = sys_.snapshot()

        reopened = System.restore(data)
        reopened.run_until(60)
        self.assertEqual(
            [(m.jid, m.task, m.deadline, m.remaining_work)
             for m in reopened.misses],
            frozen,
        )
        ex = explain_miss(reopened, frozen[0][0])
        self.assertEqual(ex.remaining_work, 1)
        self.assertTrue(ex.consistent)
        self.assertTrue(ex.work_consistent)

    def test_interrupt_arriving_exactly_at_deadline(self):
        """中断在截止拍 t=13 触发：服务拍全部落在窗口外，不占窗口归因。"""
        sys_ = System(
            tasks=inversion_tasks(h_deadline=10),
            interrupts=[InterruptSource("Tick", [13], service=2)],
            costs=COSTS,
        )
        sys_.run_until(25)
        h_job = next(j for j in sys_.jobs if j.task.name == "H")
        self.assertEqual(h_job.completed_at, 17)
        ex = explain_miss(sys_, h_job.jid)
        self.assertEqual(ex.categories["interrupt"], 0)
        self.assertEqual(ex.interrupts, {})
        # 窗口 [3,13) 外的 [13,15) ISR 拍出现在时间轴上，但不改变截止事实
        isr_segs = [s for s in sys_.trace if s.kind == "isr"]
        self.assertEqual([(s.start, s.end) for s in isr_segs], [(13, 15)])
        self.assertTrue(ex.consistent)
        self.assertTrue(ex.work_consistent)

    def test_lock_holder_preempted_no_inheritance(self):
        """关闭继承：锁持有者 L 被 M 抢占，H 被动等待，逐段对账阻塞责任。"""
        sys_ = System(tasks=inversion_tasks(h_deadline=10), costs=COSTS,
                      use_inheritance=False)
        sys_.run_until(25)
        h_job = next(j for j in sys_.jobs if j.task.name == "H")
        self.assertEqual(h_job.completed_at, 19)
        ex = explain_miss(sys_, h_job.jid)
        # 窗口 [3,13)：自身1 + 自身lock 1 + 切换3 + M 运行3 + L 运行2 = 10；
        # H 全程未被真正调度执行，M/L 占的拍都归为阻塞（优先级反转）。
        self.assertTrue(ex.consistent)
        self.assertTrue(ex.work_consistent)
        self.assertEqual(ex.preemptions, {})
        self.assertEqual(ex.inversions, {"M": 3, "L": 2})
        self.assertEqual(ex.categories["preempted"], 0)
        self.assertEqual(ex.categories["inversion"], 5)


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


class TestRuntimeModeSwitch(unittest.TestCase):
    def build(self):
        return System(
            tasks=[TaskSpec("T", 1, [run(4)], period=10)],
            costs=Costs(context_switch=0),
            modes=[
                ModeSpec("A", periods={"T": 10}, deadlines={"T": 10}),
                ModeSpec("B", periods={"T": 3}, deadlines={"T": 3}),
            ],
            initial_mode="A",
        )

    def test_immediate_switch_keeps_existing_job_promise(self):
        sys_ = self.build()
        sys_.run_until(2)
        sys_.switch_mode("B", "immediate")
        sys_.run_until(11)

        first, second = sys_.jobs
        self.assertEqual((first.mode, first.period, first.rel_deadline),
                         ("A", 10, 10))
        self.assertEqual(first.abs_deadline, 10)
        self.assertEqual(first.pc, 1)  # 已执行进度未重置
        self.assertEqual((second.release, second.mode, second.abs_deadline),
                         (10, "B", 13))
        self.assertEqual([(e.time, e.mode, e.strategy)
                          for e in sys_.mode_events],
                         [(0, "A", "initial"), (2, "B", "immediate")])
        task_segments = [
            s for s in sys_.trace if s.kind == "task" and s.jid == first.jid
        ]
        self.assertEqual(
            [(s.start, s.end, s.mode) for s in task_segments],
            [(0, 2, "A"), (2, 4, "B")],
        )

    def test_after_job_switch_waits_for_requested_job(self):
        immediate = self.build()
        immediate.run_until(2)
        immediate.switch_mode("B", "immediate")
        immediate.run_until(5)

        delayed = self.build()
        delayed.run_until(2)
        delayed.switch_mode("B", "after_job")
        self.assertEqual(delayed.mode, "A")
        delayed.run_until(3)
        self.assertEqual(delayed.mode, "A")
        delayed.run_until(4)
        self.assertEqual(delayed.mode, "B")
        delayed.run_until(5)

        self.assertEqual(
            [(e.time, e.mode, e.strategy, e.target_jid)
             for e in delayed.mode_events],
            [(0, "A", "initial", None),
             (4, "B", "after_job", 0)],
        )
        self.assertEqual(delayed.jobs[0].abs_deadline, 10)

    def test_switch_at_deadline_does_not_rewrite_old_responsibility(self):
        system = System(
            tasks=[TaskSpec("T", 1, [run(5)], period=10,
                            deadline=4)],
            costs=Costs(context_switch=0),
            modes=[ModeSpec("A"), ModeSpec("B")],
            initial_mode="A",
        )
        system.run_until(3)
        system.switch_mode("B", "immediate", at_time=4)
        system.run_until(5)

        miss = system.misses[0]
        self.assertEqual((miss.release_mode, miss.deadline_mode),
                         ("A", "B"))
        self.assertEqual(miss.remaining_work, 1)
        explanation = explain_miss(system, miss.jid)
        self.assertEqual(explanation.categories["self_run"], 4)
        self.assertEqual(set(explanation.mode_breakdown), {"A"})
        self.assertEqual(explanation.mode_breakdown["A"]["self_run"], 4)


class TestModeLockBudgetAndInheritance(unittest.TestCase):
    def build(self):
        return System(
            tasks=[
                TaskSpec("L", 1,
                         [lock("R"), run(3), unlock("R")],
                         period=100),
                TaskSpec("H", 3, [lock("R"), run(1), unlock("R")],
                         period=100, offset=2, deadline=100),
            ],
            costs=Costs(context_switch=0, lock=1, unlock=1),
            modes=[
                ModeSpec("A", lock_budgets={"R": 2}),
                ModeSpec("B", lock_budgets={"R": 100}),
            ],
            initial_mode="A",
        )

    def test_cross_mode_inheritance_uses_acquisition_mode_budget(self):
        sys_ = self.build()
        sys_.run_until(1)
        sys_.switch_mode("B", "immediate")
        sys_.run_until(3)

        low, high = sys_.jobs
        self.assertEqual(low.held, ["R"])
        self.assertEqual(high.state, "blocked")
        self.assertEqual(high.lock_request_mode, "B")
        self.assertEqual(low.boost, 2)  # 跨 A/B 的等待仍传递优先级
        self.assertEqual(sys_.resources["R"]["mode"], "A")
        self.assertEqual(sys_.resources["R"]["budget"], 2)

        sys_.run_until(6)
        self.assertEqual(low.completed_at, 6)
        self.assertEqual(high.mode, "B")
        violation, = sys_.lock_budget_violations
        self.assertEqual(
            (violation.resource, violation.task, violation.mode,
             violation.acquired_at, violation.budget,
             violation.overrun, violation.elapsed_at_violation,
             violation.released_at),
            ("R", "L", "A", 1, 2, 1, 3, 6),
        )


class TestNestedInterrupts(unittest.TestCase):
    def test_higher_priority_isr_preempts_lower_isr(self):
        sys_ = System(
            tasks=[TaskSpec("A", 1, [run(10)], period=100)],
            interrupts=[
                InterruptSource("Low", [1], service=3, priority=1),
                InterruptSource("High", [2], service=1, priority=2),
            ],
            costs=Costs(context_switch=0),
        )
        sys_.run_until(7)
        self.assertEqual(
            [(s.start, s.end, s.kind, s.detail)
             for s in sys_.trace if s.kind in ("isr", "task")],
            [
                (0, 1, "task", "A#0"),
                (1, 2, "isr", "isr:Low"),
                (2, 3, "isr", "isr:High"),
                (3, 5, "isr", "isr:Low"),
                (5, 7, "task", "A#0"),
            ],
        )


class TestModeSnapshot(unittest.TestCase):
    def test_pending_after_job_switch_resumes_on_same_timeline(self):
        def build():
            return System(
                tasks=[TaskSpec("T", 1, [run(4)], period=10)],
                costs=Costs(context_switch=0),
                modes=[ModeSpec("A"), ModeSpec("B")],
                initial_mode="A",
            )

        whole = build()
        whole.run_until(2)
        whole.switch_mode("B", "after_job")
        whole.run_until(8)

        partial = build()
        partial.run_until(2)
        partial.switch_mode("B", "after_job")
        partial.run_until(3)
        reopened = System.restore(partial.snapshot())
        reopened.run_until(8)

        self.assertEqual(segs(whole), segs(reopened))
        self.assertEqual(
            [(e.time, e.mode, e.strategy, e.target_jid)
             for e in whole.mode_events],
            [(e.time, e.mode, e.strategy, e.target_jid)
             for e in reopened.mode_events],
        )


if __name__ == "__main__":
    unittest.main()

class TestEscalation(unittest.TestCase):
    """关键程度临时升级：触发、让步、恢复、撤回。

    场景A（defer 让步）：L(优先级1,wcet6) 与 K(优先级2,offset4,wcet2)，
    K 在 t=0 请求升级（+5）。K 的作业 t=4 才释放，升级在 t=5 边界生效。
    手算：switch[0,1) L[1,4) switch[4,5) K[5,7) switch[7,8) L[8,11)
    L 在 [5,7) 被让步推迟，K 完成（升级结束）后于 t=7 恢复。
    """

    @staticmethod
    def build_a(strategy="defer"):
        sys_ = System(
            tasks=[
                TaskSpec("L", 1, [run(6)], period=100),
                TaskSpec("K", 2, [run(2)], period=100, offset=4),
            ],
            costs=COSTS,
        )
        sys_.escalate("K", 5, strategy=strategy, at_time=0)
        return sys_

    def test_defer_timeline(self):
        sys_ = self.build_a()
        sys_.run_until(12)
        self.assertEqual(
            segs(sys_),
            [
                (0, 1, "switch", "switch->L#0"),
                (1, 4, "task", "L#0"),
                (4, 5, "switch", "switch->K#1"),
                (5, 7, "task", "K#1"),
                (7, 8, "switch", "switch->L#0"),
                (8, 11, "task", "L#0"),
                (11, 12, "idle", "idle"),
            ],
        )
        assert_contiguous(self, sys_, 12)

    def test_deferral_record(self):
        sys_ = self.build_a()
        sys_.run_until(12)
        self.assertEqual(len(sys_.deferrals), 1)
        rec = sys_.deferrals[0]
        self.assertEqual((rec.jid, rec.task, rec.deferred_at, rec.resumed_at),
                         (0, "L", 5, 7))
        self.assertIn("完成", rec.reason)
        esc = sys_.escalations[0]
        self.assertEqual((esc.state, esc.applied_at, esc.ended_at),
                         ("done", 5, 7))
        k = next(j for j in sys_.jobs if j.task.name == "K")
        low = next(j for j in sys_.jobs if j.task.name == "L")
        self.assertEqual(k.esc_boost, 0)
        self.assertFalse(low.deferred)

    def test_priority_strategy_defers_nobody(self):
        """让步策略对比：priority 只抬优先级，不产生任何推迟记录。"""
        sys_ = self.build_a(strategy="priority")
        sys_.run_until(12)
        self.assertEqual(sys_.deferrals, [])
        self.assertEqual(explain_deferrals(sys_),
                         ["没有任何任务被让步推迟。"])

    def test_explain_deferrals_report(self):
        sys_ = self.build_a()
        sys_.run_until(12)
        lines = explain_deferrals(sys_)
        self.assertEqual(len(lines), 1)
        self.assertIn("L#0", lines[0])
        self.assertIn("[5, 7)", lines[0])
        self.assertIn("恢复", lines[0])

    def test_unresumed_deferral_explained(self):
        """升级一直生效时，被推迟作业必须标注“至今未恢复”及原因。"""
        sys_ = System(
            tasks=[
                TaskSpec("L", 1, [run(4)], period=100),
                TaskSpec("K", 2, [run(20)], period=100),
            ],
            costs=COSTS,
        )
        sys_.escalate("K", 5, strategy="defer", at_time=0)
        sys_.run_until(8)
        rec = sys_.deferrals[0]
        self.assertIsNone(rec.resumed_at)
        report = "\n".join(explain_deferrals(sys_))
        self.assertIn("至今", report)
        self.assertIn("仍未恢复", report)


class TestEscalationWithLock(unittest.TestCase):
    """升级任务等锁时：持有者的临界区不可暂停，放锁后才可让步。

    L(优先级1): lock R, run2, unlock R, run2；K(优先级2,offset2):
    run1, lock R, run1, unlock R。K 在 t=0 请求升级（+5, defer）。
    手算：switch[0,1) L-lock[1,2) switch[2,3) K[3,4) K-lock[4,5)(阻塞)
          switch[5,6) L[6,8) L-unlock[8,9)   <- L 持锁期间不被推迟
          switch[9,10) K-lock[10,11) K[11,12) K-unlock[12,13)
          switch[13,14) L[14,16)             <- L 放锁后才被推迟，K 完成后恢复
    """

    @staticmethod
    def build():
        sys_ = System(
            tasks=[
                TaskSpec("L", 1, [lock("R"), run(2), unlock("R"), run(2)],
                         period=100),
                TaskSpec("K", 2, [run(1), lock("R"), run(1), unlock("R")],
                         period=100, offset=2),
            ],
            costs=COSTS,
        )
        sys_.escalate("K", 5, strategy="defer", at_time=0)
        return sys_

    def test_holder_finishes_critical_section(self):
        sys_ = self.build()
        sys_.run_until(20)
        self.assertEqual(
            segs(sys_),
            [
                (0, 1, "switch", "switch->L#0"),
                (1, 2, "lock", "L#0 lock R"),
                (2, 3, "switch", "switch->K#1"),
                (3, 4, "task", "K#1"),
                (4, 5, "lock", "K#1 lock R"),
                (5, 6, "switch", "switch->L#0"),
                (6, 8, "task", "L#0"),
                (8, 9, "unlock", "L#0 unlock R"),
                (9, 10, "switch", "switch->K#1"),
                (10, 11, "lock", "K#1 lock R"),
                (11, 12, "task", "K#1"),
                (12, 13, "unlock", "K#1 unlock R"),
                (13, 14, "switch", "switch->L#0"),
                (14, 16, "task", "L#0"),
                (16, 20, "idle", "idle"),
            ],
        )
        done = {j.task.name: j.completed_at for j in sys_.jobs}
        self.assertEqual(done["K"], 13)
        self.assertEqual(done["L"], 16)
        # L 持锁期间（[1,9)）从未被推迟；放锁后 [9,13) 才被让步
        rec = sys_.deferrals[0]
        self.assertEqual((rec.deferred_at, rec.resumed_at), (9, 13))
        assert_contiguous(self, sys_, 20)

    def test_snapshot_mid_escalation(self):
        """升级生效中途保存，续跑与一次跑完逐段一致。"""
        part = self.build()
        part.run_until(9)
        restored = System.restore(part.snapshot())
        restored.run_until(20)
        whole = self.build()
        whole.run_until(20)
        self.assertEqual(segs(restored), segs(whole))
        key = lambda d: (d.jid, d.deferred_at, d.resumed_at, d.reason)
        self.assertEqual([key(d) for d in restored.deferrals],
                         [key(d) for d in whole.deferrals])


class TestEscalationDeadlineAndRevoke(unittest.TestCase):
    """response 收紧后续响应要求；撤回恢复原承诺，但既有超期事实冻结。

    K(优先级2,wcet6,相对截止20)，t=0 请求升级（response=2）。
    升级 t=1 生效，绝对截止收紧为 1+2=3；t=3 超期（已执行2拍，还差4拍）。
    t=4 撤回，截止恢复 20；MissRecord 仍记录 deadline=3。
    """

    def test_tighten_miss_revoke(self):
        sys_ = System(
            tasks=[TaskSpec("K", 2, [run(6)], period=100, deadline=20)],
            costs=COSTS,
        )
        eid = sys_.escalate("K", 3, response=2, strategy="priority",
                            at_time=0)
        sys_.run_until(4)
        job = sys_.jobs[0]
        self.assertEqual(len(sys_.misses), 1)
        miss = sys_.misses[0]
        self.assertEqual((miss.deadline, miss.remaining_work), (3, 4))
        sys_.revoke(eid)
        self.assertEqual(job.abs_deadline, 20)  # 撤回恢复原承诺
        sys_.run_until(10)
        self.assertEqual(job.completed_at, 7)
        # 既有超期事实不被撤回改写
        self.assertEqual(sys_.misses[0].deadline, 3)
        exp = explain_miss(sys_, 0)
        self.assertEqual(exp.deadline, 3)
        self.assertEqual(exp.window, 3)
        self.assertTrue(exp.consistent)
        self.assertTrue(exp.work_consistent)
        self.assertEqual(sys_.escalations[0].state, "revoked")
        self.assertEqual(sys_.escalations[0].ended_at, 4)


class TestEscalationExplainDeferred(unittest.TestCase):
    """错过截止的归因包含“被让步推迟”类别，且各项之和恒等于窗口。

    L(优先级1,wcet6,相对截止10) 被 K 的升级让步推迟 [5,7)，t=10 超期。
    窗口 [0,10)：切换3拍 + 自身5拍 + 推迟2拍 = 10拍。
    """

    def test_deferred_category(self):
        sys_ = System(
            tasks=[
                TaskSpec("L", 1, [run(6)], period=100, deadline=10),
                TaskSpec("K", 2, [run(2)], period=100, offset=4),
            ],
            costs=COSTS,
        )
        sys_.escalate("K", 5, strategy="defer", at_time=0)
        sys_.run_until(15)
        exp = explain_miss(sys_, 0)
        self.assertEqual(exp.categories["deferred"], 2)
        self.assertEqual(exp.deferrals, {0: 2})
        self.assertEqual(exp.categories["self_run"], 5)
        self.assertEqual(exp.categories["context_switch"], 3)
        self.assertTrue(exp.consistent)
        self.assertTrue(exp.work_consistent)
        self.assertIn("让步推迟", exp.report())


class TestEscalationOrdering(unittest.TestCase):
    """多个升级接连提出、升级又撤回、同时切换模式：
    每项承诺的生效范围与先后关系都保留。"""

    def test_stacked_and_revoked_with_mode_switch(self):
        modes = [
            ModeSpec("normal"),
            ModeSpec("fast", periods={"K": 50}),
        ]
        sys_ = System(
            tasks=[
                TaskSpec("L", 1, [run(30)], period=100),
                TaskSpec("K", 2, [run(10)], period=100),
            ],
            costs=COSTS,
            modes=modes,
            initial_mode="normal",
        )
        e1 = sys_.escalate("K", 3, strategy="defer", at_time=2)
        e2 = sys_.escalate("K", 6, strategy="defer", at_time=4)
        sys_.switch_mode("fast", "immediate")
        sys_.run_until(3)
        self.assertEqual(sys_.escalations[e1].state, "active")
        self.assertEqual(sys_.escalations[e1].applied_at, 2)
        sys_.run_until(5)
        k = sys_.jobs[1]
        self.assertEqual(k.esc_boost, 6)   # e2 生效后取两者最大增量
        sys_.revoke(e2)
        self.assertEqual(k.esc_boost, 3)   # e1 仍在生效
        sys_.revoke(e1)
        self.assertEqual(k.esc_boost, 0)
        # 每项承诺的生效范围与先后关系可逐项回放
        self.assertEqual(
            [(e.state, e.applied_at, e.ended_at) for e in sys_.escalations],
            [("revoked", 2, 5), ("revoked", 4, 5)],
        )
        # 模式切换事件与升级事件各自保序、互不吞没
        self.assertEqual([m.mode for m in sys_.mode_events],
                         ["normal", "fast"])
        sys_.run_until(60)
        assert_contiguous(self, sys_, 60)
