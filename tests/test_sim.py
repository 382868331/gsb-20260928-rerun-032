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
    Mode,
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


class TestModeSwitching(unittest.TestCase):
    """运行期切换工作模式。

    H(优先级2)：4 拍程序 lockR/run2/unlockR/run1；
    L(优先级1)：run8。normal 下 H 周期 10；boost 下 H 周期 6。
    模式只改变“切换之后”的新释放作业；既有作业承诺不变。
    """

    def _build(self):
        normal = Mode("normal", task_periods={"H": 10},
                      task_deadlines={"H": 10},
                      lock_budgets={"R": 5})
        boost = Mode("boost", task_periods={"H": 6},
                     task_deadlines={"H": 6},
                     lock_budgets={"R": 5})
        return System(
            tasks=[
                TaskSpec("H", 2,
                         [lock("R"), run(2), unlock("R"), run(1)],
                         period=10),
                TaskSpec("L", 1, [run(8)], period=100),
            ],
            modes=[normal, boost], initial_mode="normal",
            costs=COSTS,
        )

    def test_immediate_switch_keeps_old_promise(self):
        sys_ = self._build()
        sys_.run_until(5)
        old_h = sys_.jobs[0]
        self.assertEqual(old_h.abs_deadline, 10)
        self.assertEqual(old_h.release_mode, "normal")
        sys_.switch_mode("boost")  # t=5 立即切换
        sys_.run_until(20)
        # 旧作业承诺不被改写；尚未释放的作业从 t=5 起按 boost 周期 6
        rels = [(j.task.name, j.release, j.abs_deadline, j.release_mode)
                for j in sys_.jobs if j.task.name == "H"]
        self.assertEqual(rels[0], ("H", 0, 10, "normal"))
        self.assertEqual(rels[1], ("H", 11, 17, "boost"))
        self.assertEqual(rels[2], ("H", 17, 23, "boost"))
        change = sys_.mode_changes[-1]
        self.assertEqual(
            (change.time, change.old_mode, change.new_mode, change.strategy),
            (5, "normal", "boost", "immediate"),
        )
        # 执行到一半的作业不重来：pc/已耗工作量延续，6 拍末完成
        self.assertEqual(sys_.jobs[0].completed_at, 6)

    def test_wait_job_switch_defers_until_current_jobs_done(self):
        sys_ = self._build()
        sys_.run_until(3)
        sys_.switch_mode_after_jobs("boost")  # H、L 都未完成
        sys_.run_until(6)  # H 在 6 完成，但 L 仍在跑 -> 不切换
        self.assertEqual(sys_.mode_name, "normal")
        sys_.run_until(30)  # L 在 29 完成后自动切换；新作业按 boost
        self.assertEqual(sys_.mode_name, "boost")
        change = sys_.mode_changes[-1]
        self.assertEqual(change.strategy, "wait_job")
        self.assertEqual(change.time, 29)
        # 等待期间 H 仍按旧模式 10 拍周期释放（10、20），不延长等待
        h_rels = [j.release for j in sys_.jobs if j.task.name == "H"]
        self.assertEqual(h_rels[:3], [0, 10, 20])

    def test_strategies_produce_distinct_mode_change_time(self):
        imm = self._build()
        imm.run_until(3)
        imm.switch_mode("boost")
        imm.run_until(40)
        wait = self._build()
        wait.run_until(3)
        wait.switch_mode_after_jobs("boost")
        wait.run_until(40)
        t_imm = [c.time for c in imm.mode_changes if c.new_mode == "boost"][0]
        t_wait = [c.time for c in wait.mode_changes
                  if c.new_mode == "boost"][0]
        self.assertEqual(t_imm, 3)
        self.assertGreater(t_wait, t_imm)  # 等到当前作业集合全完成

    def test_cross_mode_lock_budget_and_inheritance(self):
        """持锁跨模式：占用按拍所处模式分别记账；继承不跨模式丢失。"""
        normal = Mode("normal", lock_budgets={"R": 2})
        boost = Mode("boost", lock_budgets={"R": 1})
        sys_ = System(
            tasks=[
                # L 持锁临界区 4 拍；t=2 释放的 H 阻塞在 R 上
                TaskSpec("L", 1,
                         [lock("R"), run(4), unlock("R")], period=100),
                TaskSpec("H", 3, [lock("R"), run(1), unlock("R")],
                         period=100, offset=2),
            ],
            modes=[normal, boost], initial_mode="normal",
            costs=COSTS,
        )
        sys_.run_until(2)  # L 已 lock R；H 在 t=3 才释放
        sys_.run_until(7)  # H lock R 阻塞：L 继承 H 优先级
        l_job = next(j for j in sys_.jobs if j.task.name == "L")
        h_job = next(j for j in sys_.jobs if j.task.name == "H")
        self.assertEqual(h_job.state, "blocked")
        self.assertEqual(l_job.eff_priority, 3)
        sys_.switch_mode("boost")  # 立即切换，L 仍持锁、H 仍等待
        sys_.run_until(20)
        # 继承压力跨模式保留：L 持续以 3 跑完临界区放锁
        self.assertTrue(h_job.completed_at is not None)
        self.assertTrue(h_job.completed_at is not None)
        # 占用按模式拆分，且新模式预算(1)更紧：boost 下必然告警
        held = l_job.lock_held_by_mode["R"]
        self.assertIn("normal", held)
        self.assertIn("boost", held)
        modes_flagged = {v.mode for v in sys_.budget_violations
                         if v.jid == l_job.jid and v.resource == "R"}
        self.assertIn("boost", modes_flagged)
        for v in sys_.budget_violations:
            self.assertGreater(v.held_in_mode, v.budget)

    def test_deadline_exactly_at_switch_responsibility_split(self):
        """切换恰落在截止拍：截止窗口不含切换后拍；报告按模式拆分。"""
        normal = Mode("normal", task_periods={"H": 10},
                      task_deadlines={"H": 5}, lock_budgets={"R": 99})
        boost = Mode("boost", task_periods={"H": 6},
                     task_deadlines={"H": 6}, lock_budgets={"R": 99})
        sys_ = System(
            tasks=[
                TaskSpec("H", 2,
                         [lock("R"), run(3), unlock("R")],
                        period=10),
                TaskSpec("L", 1, [run(8)], period=100),
            ],
            interrupts=[InterruptSource("Tick", [3], service=1)],
            modes=[normal, boost], initial_mode="normal",
            costs=Costs(context_switch=0, lock=0, unlock=0),
        )
        # H 截止为 6；安排 t=6 切换（截止拍，窗口 [0,6) 之外）
        sys_.schedule_mode_switch(5, "boost")
        sys_.run_until(12)
        h0 = sys_.jobs[0]
        self.assertEqual(h0.abs_deadline, 5)
        self.assertTrue(h0.missed)
        miss = sys_.misses[0]
        self.assertEqual(miss.deadline_mode, "normal")
        ex = explain_miss(sys_, h0.jid)
        # 窗口拍全部在 normal；boost 的拍不得计入窗口、不改写超期量
        self.assertEqual(ex.window, 5)
        self.assertTrue(ex.consistent)
        self.assertIn("normal", ex.by_mode)
        # 冻结剩余量在继续运行/切换后保持不变
        frozen = miss.remaining_work
        sys_.run_until(30)
        self.assertEqual(sys_.misses[0].remaining_work, frozen)
        self.assertEqual(explain_miss(sys_, h0.jid).remaining_work, frozen)

    def test_snapshot_during_switch_phase_replays_timeline(self):
        """任意切换阶段（等待切换未生效、刚切换、持锁中）保存恢复一致。"""
        for checkpoint in (2, 4, 6, 9):
            whole = self._build()
            whole.run_until(3)
            whole.switch_mode_after_jobs("boost")
            whole.run_until(40)

            part = self._build()
            part.run_until(3)
            part.switch_mode_after_jobs("boost")
            part.run_until(checkpoint)
            data = part.snapshot()
            resumed = System.restore(data)
            resumed.run_until(40)
            self.assertEqual(segs(whole), segs(resumed))
            self.assertEqual(
                [(c.time, c.old_mode, c.new_mode, c.strategy)
                 for c in whole.mode_changes],
                [(c.time, c.old_mode, c.new_mode, c.strategy)
                 for c in resumed.mode_changes],
            )
            self.assertEqual(
                [(v.jid, v.resource, v.mode, v.time, v.held_in_mode)
                 for v in whole.budget_violations],
                [(v.jid, v.resource, v.mode, v.time, v.held_in_mode)
                 for v in resumed.budget_violations],
            )
            done_w = {j.jid: j.completed_at for j in whole.jobs}
            done_r = {j.jid: j.completed_at for j in resumed.jobs}
            self.assertEqual(done_w, done_r)


class TestNestedInterrupts(unittest.TestCase):
    """高优先级中断可抢占嵌套低优先级中断；trace 逐拍可见进出。"""

    def test_nested_isr_timeline(self):
        sys_ = System(
            tasks=[TaskSpec("A", 1, [run(8)], period=100)],
            interrupts=[
                InterruptSource("Lo", [2], service=4, priority=1),
                InterruptSource("Hi", [4], service=2, priority=2),
            ],
            costs=COSTS,
        )
        sys_.run_until(12)
        kinds = [(s.start, s.end, s.kind, s.detail) for s in sys_.trace]
        # A[1,2) 后 Lo 服务；t=4 Hi 抢占；Hi 结束后回到 Lo 剩余拍
        self.assertIn((2, 4, "isr", "isr:Lo"), kinds)
        self.assertIn((4, 6, "isr", "isr:Hi"), kinds)
        # Lo 总 4 拍：[2,4) 两拍 + Hi 后两拍
        self.assertIn((6, 8, "isr", "isr:Lo"), kinds)
        lo_ticks = sum(e - b for b, e, k, d in kinds if d == "isr:Lo")
        hi_ticks = sum(e - b for b, e, k, d in kinds if d == "isr:Hi")
        self.assertEqual(lo_ticks, 4)
        self.assertEqual(hi_ticks, 2)
        assert_contiguous(self, sys_, 12)

    def test_lower_priority_waits_until_isr_done(self):
        sys_ = System(
            tasks=[TaskSpec("A", 1, [run(8)], period=100)],
            interrupts=[
                InterruptSource("Hi", [2], service=3, priority=3),
                InterruptSource("Lo", [3], service=2, priority=1),
            ],
            costs=COSTS,
        )
        sys_.run_until(10)
        kinds = [(s.start, s.kind, s.detail) for s in sys_.trace]
        # Lo 在 t=3 触发但 Hi 在服务：必须等 Hi 结束，不能插入
        hi_span = [s for s in sys_.trace if s.detail == "isr:Hi"]
        lo_span = [s for s in sys_.trace if s.detail == "isr:Lo"]
        self.assertEqual(hi_span[0].start, 2)
        self.assertEqual(lo_span[0].start, hi_span[-1].end)


if __name__ == "__main__":
    unittest.main()
