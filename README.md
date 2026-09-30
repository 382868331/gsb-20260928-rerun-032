# 嵌入式任务仿真系统（第一版）

纯 Python、零依赖的离散时间仿真器：虚拟处理器 + 任务/中断模型 +
可抢占固定优先级调度，所有机制开销都落在时间线上。

## 模型

- **时间**：整数“拍”。每拍 CPU 只做一件事：执行任务、执行 ISR、
  上下文切换、lock/unlock 系统调用，或空转。
- **任务**（`TaskSpec`）：静态优先级（大者高）、周期、相对截止、
  首次释放偏移，以及 `program`——由 `run(n)` / `lock(R)` /
  `unlock(R)` 组成的指令步序列，WCET 为 run 步之和。
- **中断**（`InterruptSource`）：指定触发时刻，进入/服务/退出
  三段耗时都占用 CPU，不计入任何任务。
- **开销**（`Costs`）：上下文切换、lock、unlock 的系统调用代价，
  全部以拍计并出现在执行记录里。

## 优先级反转：采用的继承规则

共享资源为互斥锁，采用**立即优先级继承**（immediate/basic
priority inheritance）：任务阻塞在资源上时，持有者优先级立即提升为
所有等待者（含传递等待，迭代到不动点）的最高有效优先级；
释放资源后恢复。`System(use_inheritance=False)` 可关闭用于对比。

## 逐段执行记录

`System.trace` 是 `[start, end)` 半开区间的段序列，段类型：
`task` / `lock` / `unlock` / `switch` / `isr` / `idle`。
记录连续覆盖整个仿真区间，可逐拍手算核对（见 `tests/` 注释）。

## 错过截止时间的归因

`explain_miss(sys, jid)` 把该作业 `[release, deadline)` 窗口内的
每一拍归类：自身执行、自身系统调用、上下文切换、中断、被高优先级
抢占、被低优先级阻塞（优先级反转）、空转；各项之和恒等于窗口长度，
并输出中文报告。截止瞬间的剩余工作量冻结在 `MissRecord` 中：
作业即使在截止之后才补完，报告仍以截止时刻的状态解释（不会把
“还差一拍”事后改成“剩余零拍”）。报告同时给出工作量对账恒等式：
窗口内自身执行拍数 + 截止瞬间剩余拍数 = WCET。
演示：`python3 examples/demo.py`。

多把锁的嵌套等待链（H 等 R、持 R 的 M 又等 Q）同样逐拍建模：
优先级继承沿等待关系迭代传播到不动点（L 经 M 借到 H 的优先级），
放锁/完成后逐级恢复；关闭继承时整条链上不发生任何提升。

## 保存与继续

`sys.snapshot()` / `System.restore(data)`（或 `save/load` 文件）
序列化完整运行态；测试验证断点续跑与一次跑完的执行记录逐段一致。

## 为功耗模型预留的扩展点

- `TaskSpec.attrs` 与 `Job.record` 是开放字典：将来挂功耗参数
  （频率/电压档、动态/静态功耗系数）和逐段能耗采样，**任务记录
  结构本身不需要重新设计**。
- 功耗按 trace 段结算即可：每段已知执行主体与时长。

## 运行期工作模式

`Mode(name, task_periods=, task_deadlines=, lock_budgets=)` 描述一套
周期/相对截止/锁占用预算。`System(modes=[...], initial_mode=...)`
启用后，trace 每一段都带 `mode` 标注。

- **立即切换** `switch_mode(name)`：在当前拍边界生效；已经释放的作业
  （执行一半、阻塞等锁）不重来——程序步、已耗时间、锁状态、释放时的
  截止承诺全部保留；尚未到点的释放按新模式周期/截止重排。作业记录
  `release_mode`/`deadline_mode`；`Job.lock_held_by_mode` 把每把锁的
  持有拍按“拍所处模式”分桶。
- **等当前作业结束** `switch_mode_after_jobs(name)`：只等登记请求时
  已存在且未完成的作业（等待期间新释放的作业不延长等待），全部完成后
  自动切换，`ModeChange.strategy` 为 `wait_job`。
- **预定切换** `schedule_mode_switch(t, name)`：到点立即切换，便于对比；
  切换恰好落在截止拍时该拍属新模式，窗口仍为 `[release, deadline)`
  半开区间，截止承诺按作业的 `deadline_mode` 冻结，历史超期量不改写。
- **跨模式锁继承**：持锁/等待关系跨切换保留，继承继续沿等待链传播；
  锁占用预算按当前模式逐拍累计，越界冻结一条
  `BudgetViolation(jid, resource, mode, time, held_in_mode, budget)`，
  同一 (作业, 锁, 模式) 只在首次越界告警一次。
- **中断嵌套**：`InterruptSource.priority` 高者抢占正在服务的低者，
  服务结束后回到被抢占 ISR 继续；低者必须等高者结束。
- **归因报告**：`explain_miss` 在窗口跨模式时给出“切换前后责任拆分”，
  各模式的自身执行/切换/中断/抢占/阻塞拍数分别合计，总和仍恒等于
  窗口长度；`MissRecord.deadline_mode` 标明截止承诺依据的模式。
- **任意阶段保存恢复**：等待切换未生效、刚切换、持锁中等任何时刻
  `snapshot()/restore()` 后续跑，trace、`mode_changes`、
  `budget_violations`、作业完成时刻均与一次跑完逐拍一致。

## 布局

- `embsim/model.py` — 任务/中断/资源/开销的静态模型
- `embsim/system.py` — 虚拟处理器、调度器、继承、模式切换、中断嵌套、快照
- `embsim/explain.py` — 错过截止时间的归因解释
- `tests/test_sim.py` — 手算核对的小例子（`python3 -m unittest discover -s tests`）
- `examples/demo.py` — 端到端演示
