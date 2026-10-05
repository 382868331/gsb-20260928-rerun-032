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
    CriticalityCommitment,
    Costs,
    InterruptSource,
    ModeSpec,
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
        sys_.run_until(4)

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


class TestCriticalityEscalation(unittest.TestCase):
    """临时提高关键程度 + 让步策略。

    L(优先级1)：lock R, run3, unlock R, run2（临界区后还有可暂停工作）
    M(优先级2)：run4
    H(优先级3)：run1, lock R, run1, unlock R（offset=4）
    开销：切换 0、lock/unlock 各 1。
    t=4 升级 H 到关键级别 5（defer 策略），t=12 撤回。
    """

    def build(self):
        return System(
            tasks=[
                TaskSpec("L", 1,
                         [lock("R"), run(3), unlock("R"), run(2)],
                         period=100),
                TaskSpec("M", 2, [run(4)], period=100, offset=1),
                TaskSpec("H", 3,
                         [run(1), lock("R"), run(1), unlock("R")],
                         period=100, offset=4, deadline=30),
            ],
            costs=Costs(context_switch=0, lock=1, unlock=1),
        )

    def test_defer_preserves_work_and_resources(self):
        sys_ = self.build()
        sys_.escalate(
            CriticalityCommitment("c1", "H", priority=5,
                                  deadline=8, strategy="defer"),
            at_time=4,
        )
        sys_.run_until(9)
        jobs = {j.task.name: j for j in sys_.jobs}
        # M 已执行 3 拍（[1,4)），被延后时进度不清零
        m_job = jobs["M"]
        self.assertEqual(m_job.state, "deferred")
        self.assertEqual(m_job.deferred_by, "c1")
        self.assertEqual(m_job.remaining_work(), 1)
        # L 仍持有 R：持锁临界区部分必须继续，不能因让步卡死锁
        self.assertEqual(jobs["L"].held, ["R"])
        self.assertEqual(jobs["L"].state, "running")
        # L 借到升级后 H 的关键级别（继承压力来自承诺级别 5）
        self.assertEqual(sys_._job_crit_level(jobs["L"]), 5)

        sys_.withdraw("c1", at_time=12)
        sys_.run_until(13)
        jobs = {j.task.name: j for j in sys_.jobs}
        self.assertEqual(jobs["M"].state, "ready")
        self.assertIsNone(jobs["M"].deferred_by)
        # 让步明细：M 从 4 推迟到 12；L 临界区后剩余工作 9->12
        timeline = sys_.concession_timeline()
        self.assertEqual(
            [(e.time, e.task, e.resumed_at)
             for e in timeline],
            [(4, "L", 5), (4, "M", 12), (9, "L", 12)],
        )
        self.assertIn("锁", timeline[0].resume_reason)
        # M 延后期间已消耗的 3 拍没有消失：[13,14) 补完最后 1 拍
        sys_.run_until(14)
        m_job = next(j for j in sys_.jobs if j.task.name == "M")
        self.assertEqual(m_job.completed_at, 14)

    def test_lock_holder_runs_critical_section_then_defers_rest(self):
        sys_ = self.build()
        sys_.escalate(
            CriticalityCommitment("c1", "H", priority=5,
                                  strategy="defer"),
            at_time=4,
        )
        sys_.run_until(12)
        # 时间轴：临界区 run3 在 [6,9) 连续跑完并放锁 [9,10)，
        # H 获得锁跑完；L 放锁后的 run2 被推迟，[10,12) 不属于 L
        kinds = [(s.start, s.end, s.kind, s.detail)
                 for s in sys_.trace
                 if s.kind in ("task", "lock", "unlock")]
        self.assertIn((6, 9, "task", "L#0"), kinds)
        self.assertIn((9, 10, "unlock", "L#0 unlock R"), kinds)
        self.assertIn((11, 12, "task", "H#2"), kinds)
        l_job = next(j for j in sys_.jobs if j.task.name == "L")
        self.assertEqual(l_job.state, "deferred")
        # pc 已越过 unlock，剩余正好是临界区后的 2 拍
        self.assertEqual(l_job.pc, 3)
        self.assertEqual(l_job.remaining_work(), 2)
        self.assertEqual(l_job.held, [])

    def test_inherit_only_never_defers(self):
        sys_ = self.build()
        sys_.escalate(
            CriticalityCommitment("c1", "H", priority=5,
                                  strategy="inherit_only"),
            at_time=4,
        )
        sys_.run_until(16)
        self.assertFalse(
            any(j.state == "deferred" for j in sys_.jobs)
        )
        self.assertEqual(sys_.concession_events, [])

    def test_compare_strategies_shows_different_delay(self):
        sys_ = self.build()
        sys_.run_until(4)
        results = sys_.compare_concessions(
            CriticalityCommitment("c1", "H", priority=5,
                                  deadline=20, strategy="defer"),
            horizon=16,
        )
        self.assertEqual(results["inherit_only"]["concessions"], [])
        deferred_tasks = {
            c["task"] for c in results["defer"]["concessions"]
        }
        self.assertEqual(deferred_tasks, {"L", "M"})
        # 比较不改变原系统状态
        self.assertEqual(sys_.now, 4)

    def test_escalation_tightens_only_newly_released_jobs(self):
        sys_ = System(
            tasks=[TaskSpec("H", 3, [run(2)], period=10,
                            deadline=100)],
            costs=Costs(context_switch=0),
        )
        sys_.run_until(2)  # 第一个作业已释放，截止 100
        sys_.escalate(
            CriticalityCommitment("c1", "H", priority=5,
                                  deadline=3, strategy="defer")
        )
        sys_.run_until(11)
        first, second = sys_.jobs
        # 已释放作业的承诺不改：截止仍是 100
        self.assertEqual(first.abs_deadline, 100)
        self.assertIsNone(first.release_commitment)
        # 升级期间新释放的作业（t=10）采用更紧的相对截止
        self.assertEqual(second.release, 10)
        self.assertEqual(second.abs_deadline, 13)
        self.assertEqual(second.rel_deadline, 3)
        self.assertEqual(second.release_commitment, "c1")

    def test_consecutive_escalations_and_withdrawals_keep_order(self):
        sys_ = System(
            tasks=[TaskSpec("A", 1, [run(10)], period=100),
                   TaskSpec("B", 2, [run(10)], period=100)],
            costs=Costs(context_switch=0),
        )
        # 同一拍提出：升级 A、撤回 A、再升级 A——按提出先后生效
        sys_.escalate(CriticalityCommitment("c", "A", priority=5),
                      at_time=2)
        sys_.withdraw("c", at_time=2)
        sys_.escalate(CriticalityCommitment("c", "A", priority=5),
                      at_time=2)
        sys_.run_until(3)
        self.assertEqual(
            [(e.time, e.seq, e.kind) for e in sys_.commitment_events],
            [(2, 0, "escalate"), (2, 1, "withdraw"),
             (2, 2, "escalate")],
        )
        self.assertIn("c", sys_.active_commitments)

    def test_mode_switch_and_escalation_coexist(self):
        sys_ = System(
            tasks=[TaskSpec("T", 1, [run(4)], period=10)],
            costs=Costs(context_switch=0),
            modes=[ModeSpec("A"), ModeSpec("B")],
            initial_mode="A",
        )
        sys_.switch_mode("B", "immediate", at_time=2)
        sys_.escalate(
            CriticalityCommitment("c", "T", priority=4), at_time=2
        )
        sys_.run_until(3)
        self.assertEqual(sys_.mode, "B")
        self.assertIn("c", sys_.active_commitments)
        self.assertEqual(
            [(e.time, e.mode) for e in sys_.mode_events],
            [(0, "A"), (2, "B")],
        )

    def test_snapshot_resume_same_timeline_with_commitments(self):
        def build():
            s = self.build()
            s.escalate(
                CriticalityCommitment("c1", "H", priority=5,
                                      deadline=20, strategy="defer"),
                at_time=4,
            )
            s.withdraw("c1", at_time=12)
            return s

        whole = build()
        whole.run_until(20)
        part = build()
        part.run_until(9)
        reopened = System.restore(part.snapshot())
        reopened.run_until(20)
        self.assertEqual(segs(whole), segs(reopened))
        self.assertEqual(
            [(e.time, e.seq, e.kind, e.name)
             for e in whole.commitment_events],
            [(e.time, e.seq, e.kind, e.name)
             for e in reopened.commitment_events],
        )
        self.assertEqual(
            [(c.time, c.task, c.jid, c.resumed_at)
             for c in whole.concession_events],
            [(c.time, c.task, c.jid, c.resumed_at)
             for c in reopened.concession_events],
        )

    def test_prior_miss_not_rewritten_by_later_escalation(self):
        sys_ = System(
            tasks=[TaskSpec("T", 1, [run(5)], period=10, deadline=3)],
            costs=Costs(context_switch=0),
        )
        sys_.run_until(4)  # t=3 截止瞬间还差 2 拍
        sys_.escalate(
            CriticalityCommitment("c", "T", priority=5, deadline=100)
        )
        sys_.run_until(8)
        miss = sys_.misses[0]
        self.assertEqual(miss.remaining_work, 2)
        ex = explain_miss(sys_, miss.jid)
        self.assertEqual(ex.remaining_work, 2)
        self.assertTrue(ex.work_consistent)

    def test_explain_attributes_deferred_ticks_as_concession(self):
        sys_ = System(
            tasks=[
                TaskSpec("L", 1,
                         [lock("R"), run(3), unlock("R"), run(2)],
                         period=100),
                TaskSpec("M", 2, [run(4)], period=100, offset=1,
                         deadline=7),
                TaskSpec("H", 3,
                         [run(1), lock("R"), run(1), unlock("R")],
                         period=100, offset=4, deadline=30),
            ],
            costs=Costs(context_switch=0, lock=1, unlock=1),
        )
        # M 截止 8，升级在 4 生效后 M 被延后；超期归因应含让步拍
        sys_.escalate(
            CriticalityCommitment("c1", "H", priority=5,
                                  strategy="defer"),
            at_time=4,
        )
        sys_.run_until(12)
        m_job = next(j for j in sys_.jobs if j.task.name == "M")
        self.assertTrue(m_job.missed)
        ex = explain_miss(sys_, m_job.jid)
        # 窗口 [1,8)：已执行 [1,4) 3 拍；[4,8) 处于 deferred 态
        self.assertEqual(ex.categories["self_run"], 3)
        self.assertEqual(ex.categories["conceded"], 4)
        self.assertTrue(ex.consistent)
        self.assertIn("让步策略被推迟", ex.report())

    def test_pending_deferrals_explained(self):
        sys_ = self.build()
        sys_.escalate(
            CriticalityCommitment("c1", "H", priority=5,
                                  strategy="defer"),
            at_time=4,
        )
        sys_.run_until(11)
        pending = sys_.pending_deferrals()
        tasks = {p["task"]: p for p in pending}
        self.assertEqual(set(tasks), {"L", "M"})
        self.assertEqual(tasks["M"]["remaining_work"], 1)
        self.assertEqual(tasks["L"]["commitment"], "c1")
        self.assertIsNotNone(tasks["L"]["deferred_at"])


class TestMultiSourceDeferralAndCycles(unittest.TestCase):
    """多来源让步、嵌套等待链分别撤回、真实资源环与分类汇报。"""

    def test_snapshot_resume_with_new_workload_and_timing(self):
        """改变工作量与升级时刻：连续运行 vs 中途保存重开逐拍一致。"""
        def build():
            s = System(
                tasks=[
                    TaskSpec("X", 1,
                             [run(2), lock("R"), run(2), unlock("R"), run(1)],
                             period=50),
                    TaskSpec("Y", 2, [run(3)], period=50, offset=2),
                    TaskSpec("Z", 3,
                             [run(1), lock("R"), run(1), unlock("R")],
                             period=50, offset=5, deadline=40),
                ],
                costs=Costs(context_switch=0, lock=1, unlock=1),
            )
            s.escalate(
                CriticalityCommitment("cz", "Z", priority=6,
                                      deadline=10, strategy="defer"),
                at_time=5,
            )
            s.withdraw("cz", at_time=14)
            return s

        whole = build()
        whole.run_until(24)
        part = build()
        part.run_until(10)
        reopened = System.restore(part.snapshot())
        reopened.run_until(24)

        self.assertEqual(segs(whole), segs(reopened))
        self.assertEqual(
            [(e.time, e.seq, e.kind, e.name) for e in whole.commitment_events],
            [(e.time, e.seq, e.kind, e.name)
             for e in reopened.commitment_events],
        )
        self.assertEqual(
            [(c.time, c.task, c.jid, c.commitment, c.resumed_at)
             for c in whole.concession_events],
            [(c.time, c.task, c.jid, c.commitment, c.resumed_at)
             for c in reopened.concession_events],
        )
        # 临时承诺数量：一次升级 + 一次撤回，结束后无生效承诺
        self.assertEqual(len(whole.commitment_events), 2)
        self.assertEqual(whole.active_commitments, {})
        # 让步确实发生过（不是跳过让步路径）
        self.assertTrue(whole.concession_events)

    def _build_chain(self):
        """L 持 Q；M 持 R 等 Q；H 等 R：三层等待链。"""
        return System(
            tasks=[
                TaskSpec("L", 1, [lock("Q"), run(2), unlock("Q")],
                         period=100),
                TaskSpec("M", 2,
                         [lock("R"), run(1), lock("Q"), run(1),
                          unlock("Q"), unlock("R")],
                         period=100, offset=2),
                TaskSpec("H", 3,
                         [run(1), lock("R"), run(1), unlock("R")],
                         period=100, offset=4, deadline=60),
            ],
            costs=Costs(context_switch=0, lock=1, unlock=1),
        )

    def test_withdraw_outer_request_restores_constraints(self):
        """撤回外层（H 的）升级请求：链上保护解除，继承压力回落。"""
        sys_ = self._build_chain()
        sys_.escalate(
            CriticalityCommitment("outer", "H", priority=8,
                                  strategy="defer"),
            at_time=6,
        )
        sys_.run_until(7)
        jobs = {j.task.name: j for j in sys_.jobs}
        # H 等 R、M 等 Q：持锁链上的 M、L 都被豁免，先把锁放出
        self.assertNotEqual(jobs["M"].state, "deferred")
        self.assertNotEqual(jobs["L"].state, "deferred")
        # 升级后的关键级别沿等待链传播到最底层持有者
        self.assertEqual(sys_._job_crit_level(jobs["L"]), 8)
        self.assertEqual(sys_._job_crit_level(jobs["M"]), 8)

        sys_.withdraw("outer", at_time=7)
        sys_.run_until(8)
        jobs = {j.task.name: j for j in sys_.jobs}
        # 撤回后只剩普通优先级继承：L 借到 H 的静态级别 3
        self.assertEqual(sys_._job_crit_level(jobs["L"]), 3)
        self.assertEqual(
            [(e.kind, e.name) for e in sys_.commitment_events],
            [("escalate", "outer"), ("withdraw", "outer")],
        )
        # 链最终解开，全部作业正常完成，已完成工作不回滚
        sys_.run_until(20)
        self.assertTrue(all(j.state == "done" for j in sys_.jobs))

    def test_withdraw_inner_request_restores_constraints(self):
        """撤回内层（M 的）升级请求：底层持有者失去升级压力。"""
        sys_ = self._build_chain()
        sys_.escalate(
            CriticalityCommitment("inner", "M", priority=7,
                                  strategy="defer"),
            at_time=6,
        )
        sys_.run_until(7)
        jobs = {j.task.name: j for j in sys_.jobs}
        # M 等 Q：L 在持锁链上被豁免；M 自身是承诺目标不被延后
        self.assertNotEqual(jobs["L"].state, "deferred")
        self.assertEqual(sys_._job_crit_level(jobs["L"]), 7)

        sys_.withdraw("inner", at_time=7)
        sys_.run_until(8)
        jobs = {j.task.name: j for j in sys_.jobs}
        # 内层请求消失后恢复普通继承（H 经 M 传下来的级别 3）
        self.assertEqual(sys_._job_crit_level(jobs["L"]), 3)
        sys_.run_until(20)
        self.assertTrue(all(j.state == "done" for j in sys_.jobs))

    def test_shared_sources_withdraw_one_keeps_others(self):
        """共享来源：两个承诺同时延后同一作业，分别撤回。"""
        sys_ = System(
            tasks=[
                TaskSpec("A", 1, [run(20)], period=100),
                TaskSpec("B", 3, [run(2)], period=100, offset=2),
                TaskSpec("C", 4, [run(2)], period=100, offset=3),
            ],
            costs=Costs(context_switch=0),
        )
        sys_.escalate(CriticalityCommitment("cB", "B", priority=8,
                                            strategy="defer"),
                      at_time=2)
        sys_.escalate(CriticalityCommitment("cC", "C", priority=9,
                                            strategy="defer"),
                      at_time=3)
        sys_.run_until(5)
        a = next(j for j in sys_.jobs if j.task.name == "A")
        # A 已执行 [0,2) 两拍，延后不清零；两个来源都登记在案
        self.assertEqual(a.state, "deferred")
        self.assertEqual(a.deferred_by_sources, ["cB", "cC"])
        self.assertEqual(a.deferred_by, "cB")
        self.assertEqual(a.remaining_work(), 18)

        # 撤回一个来源：另一个仍有效，让步继续
        sys_.withdraw("cB", at_time=10)
        sys_.run_until(11)
        self.assertEqual(a.state, "deferred")
        self.assertEqual(a.deferred_by_sources, ["cC"])

        # 全部来源消失后才恢复原调度约束
        sys_.withdraw("cC", at_time=12)
        sys_.run_until(13)
        self.assertEqual(a.state, "ready")
        self.assertEqual(a.deferred_by_sources, [])
        self.assertIsNone(a.deferred_by)
        sys_.run_until(32)
        # 已完成工作不回滚：A 恢复后 B 先补完被 cC 延后的最后 1 拍，
        # A 再在 [13,31) 补完剩余 18 拍
        self.assertEqual(a.completed_at, 31)
        events = [e for e in sys_.concession_events if e.jid == a.jid]
        self.assertEqual(
            [(e.commitment, e.time, e.resumed_at) for e in events],
            [("cB", 2, 10), ("cC", 3, 12)],
        )


def manual_cycle_system():
    return System(
        tasks=[
            TaskSpec("A", 2,
                     [lock("P"), run(1), lock("Q"), run(1),
                      unlock("Q"), unlock("P")],
                     period=100),
            TaskSpec("B", 3,
                     [lock("Q"), run(1), lock("P"), run(1),
                      unlock("P"), unlock("Q")],
                     period=100, offset=1),
            TaskSpec("D", 1, [run(10)], period=100),
            TaskSpec("X", 1, [run(1)], period=100, offset=20),
        ],
        costs=Costs(context_switch=0, lock=1, unlock=1),
    )


class TestManualCycleIntervention(unittest.TestCase):
    def test_preview_does_not_change_schedule_and_abort_is_confirmed(self):
        sys_ = manual_cycle_system()
        sys_.escalate(CriticalityCommitment("cD", "D", priority=8,
                                            strategy="defer"),
                      at_time=6)
        sys_.escalate(CriticalityCommitment("cX", "X", priority=9,
                                            strategy="defer"),
                      at_time=6)
        sys_.run_until(6)
        a = next(j for j in sys_.jobs if j.task.name == "A")
        b = next(j for j in sys_.jobs if j.task.name == "B")

        preview = sys_.prepare_resource_cycle_intervention(
            a.jid, action="abort", intervention_id="abort-a")
        self.assertEqual(preview["resources_released"], ["P"])
        self.assertEqual(preview["completed_work_preserved"]["executed_work"], 1)
        self.assertEqual(preview["completed_work_preserved"]["remaining_work"], 1)
        self.assertEqual(preview["affected_jobs"], [a.jid, b.jid])

        # 预演不改变当前调度：环、持有资源和时钟仍保持确认前状态
        self.assertEqual(sys_.now, 6)
        self.assertEqual(a.state, "blocked")
        self.assertEqual(b.state, "blocked")
        self.assertEqual(a.held, ["P"])
        self.assertEqual(sys_.manual_intervention_history(), [])

        # 未确认预演保存重开后仍是未确认状态，不会自动处置
        reopened = System.restore(sys_.snapshot())
        pending = reopened.pending_resource_cycle_interventions()
        self.assertEqual(len(pending), 1)
        self.assertFalse(pending[0]["confirmed"])
        self.assertEqual(pending[0]["status"], "unconfirmed")

        reopened.confirm_resource_cycle_intervention("abort-a")
        reopened_b = next(j for j in reopened.jobs if j.task.name == "B")
        reopened_a = next(j for j in reopened.jobs if j.task.name == "A")
        self.assertEqual(reopened_a.state, "aborted")
        self.assertEqual(reopened_a.held, [])
        self.assertEqual(reopened_a.remaining_work(), 1)
        self.assertIsNone(reopened.resources["P"]["holder"])
        self.assertEqual(reopened.resources["P"]["waiters"], [])
        self.assertEqual(reopened.resources["Q"]["waiters"], [])
        self.assertEqual(reopened.wait_for_graph(), {})
        self.assertEqual(reopened_b.state, "deferred")
        self.assertEqual(reopened_b.deferred_by_sources, ["cD", "cX"])

        # 人工处置不清空多条临时承诺依据；撤一条仍受另一条约束
        reopened.withdraw("cD", at_time=7)
        reopened.run_until(8)
        self.assertEqual(reopened_b.state, "deferred")
        self.assertEqual(reopened_b.deferred_by_sources, ["cX"])
        reopened.withdraw("cX", at_time=8)
        reopened.run_until(12)
        self.assertEqual(reopened_b.state, "done")

        # 重复确认不重复释放资源；审计记录可追溯实际作业和资源
        again = reopened.confirm_resource_cycle_intervention("abort-a")
        self.assertEqual(again["resources_released"], ["P"])
        self.assertEqual(len(reopened.manual_intervention_history()), 1)
        record = reopened.manual_intervention_history()[0]
        self.assertEqual(record["target"], {"jid": a.jid, "task": "A"})
        self.assertEqual(record["affected_jobs"], [a.jid, b.jid])
        self.assertEqual(record["commitment_basis_preserved"], ["cD", "cX"])

    def test_timeout_requires_confirmation_and_fires_once(self):
        sys_ = manual_cycle_system()
        sys_.run_until(6)
        a = next(j for j in sys_.jobs if j.task.name == "A")
        preview = sys_.prepare_resource_cycle_intervention(
            a.jid, action="timeout", timeout=2,
            intervention_id="wait-a")
        self.assertEqual(preview["execute_at"], 8)
        self.assertEqual(preview["status"], "applied")

        # 未确认超时不会触发
        sys_.run_until(9)
        self.assertEqual(a.state, "blocked")
        self.assertEqual(a.held, ["P"])
        self.assertEqual(sys_.manual_intervention_history(), [])

        # 保存重开后仍需确认；确认后下一边界触发
        reopened = System.restore(sys_.snapshot())
        reopened_a = next(j for j in reopened.jobs if j.task.name == "A")
        self.assertEqual(
            reopened.pending_resource_cycle_interventions()[0]["status"],
            "unconfirmed",
        )
        armed = reopened.confirm_resource_cycle_intervention("wait-a")
        self.assertEqual(armed["status"], "armed")
        reopened.run_until(10)
        self.assertEqual(reopened_a.state, "aborted")
        self.assertEqual(reopened_a.held, [])

        reopened.run_until(12)
        self.assertEqual(len(reopened.manual_intervention_history()), 1)
        self.assertEqual(reopened.pending_resource_cycle_interventions(), [])

    def test_real_resource_cycle_reported_separately(self):
        """真实资源环：保留持有关系、说明优先级无法解决、独立任务不受拖累。"""
        sys_ = System(
            tasks=[
                TaskSpec("A", 2,
                         [lock("P"), run(1), lock("Q"), run(1),
                          unlock("Q"), unlock("P")],
                         period=100),
                TaskSpec("B", 3,
                         [lock("Q"), run(1), lock("P"), run(1),
                          unlock("P"), unlock("Q")],
                         period=100, offset=1),
                TaskSpec("D", 1, [run(3)], period=100),
            ],
            costs=Costs(context_switch=0, lock=1, unlock=1),
        )
        # 临时承诺照常建立与撤回（inherit_only，不干扰环的形成）
        sys_.escalate(CriticalityCommitment("cd", "D",
                                            strategy="inherit_only"),
                      at_time=0)
        sys_.withdraw("cd", at_time=6)
        sys_.run_until(12)

        jobs = {j.task.name: j for j in sys_.jobs}
        # 环上双方互相持有对方所需资源，永久阻塞；已完成工作保留
        self.assertEqual(jobs["A"].state, "blocked")
        self.assertEqual(jobs["B"].state, "blocked")
        self.assertEqual(jobs["A"].held, ["P"])
        self.assertEqual(jobs["B"].held, ["Q"])
        self.assertEqual(jobs["A"].remaining_work(), 1)
        self.assertEqual(jobs["B"].remaining_work(), 1)
        # 独立任务 D 不受无关环拖累，正常完成
        self.assertEqual(jobs["D"].state, "done")
        self.assertEqual(jobs["D"].completed_at, 9)

        # 等待环：任务身份与真实持有关系
        cycles = sys_.wait_cycles()
        self.assertEqual(len(cycles), 1)
        self.assertEqual(sorted(cycles[0]["tasks"]), ["A", "B"])
        holds = {m["task"]: m["holds"] for m in cycles[0]["members"]}
        self.assertEqual(holds, {"A": ["P"], "B": ["Q"]})
        self.assertIn("优先级", cycles[0]["explanation"])

        # 正常完成、无法解除的等待、错误分别报告
        report = sys_.run_report()
        self.assertEqual([c["task"] for c in report["completed"]], ["D"])
        self.assertEqual(len(report["unresolvable_waits"]), 1)
        self.assertEqual(report["errors"],
                         {"missed_deadlines": [],
                          "lock_budget_violations": []})
        # 临时承诺数量：一次升级一次撤回，无残留
        self.assertEqual(len(sys_.commitment_events), 2)
        self.assertEqual(sys_.active_commitments, {})
        # 原有调度入口仍可用
        sys_.run(2)
        self.assertEqual(sys_.now, 14)


if __name__ == "__main__":
    unittest.main()
