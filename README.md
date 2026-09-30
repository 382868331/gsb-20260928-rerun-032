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
  三段耗时都占用 CPU，不计入任何任务；`priority` 较高的 ISR 可在
  整数拍边界嵌套抢占低优先级 ISR。
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

## 运行期工作模式

`ModeSpec(name, periods=..., deadlines=..., lock_budgets=...)`
描述一组任务周期、相对截止和锁占用预算。构造 `System` 时传入
`modes=[...]` 与 `initial_mode`，运行中可调用：

- `switch_mode(name, "immediate")`：下一拍边界生效，不打断或重跑当前指令。
- `switch_mode(name, "after_job")`：等待请求时的当前作业完成后的拍末生效。

`switch_mode(..., at_time=t)` 可把立即切换预先调度到指定整数边界，
用于复现“恰好在截止时刻切换”的场景。

模式切换只影响切换后释放的新作业。已有作业在其 `Job` 中冻结释放模式、
周期、相对截止和绝对截止；正在执行的 `pc/step_left` 与阻塞状态原样保留。
阻塞后获得锁的作业使用其发起 lock 请求时的模式预算；持有者跨模式放锁
不会按新模式改写账。超预算事件记录资源、作业、获得模式、首次超期时刻、
首次超期量和释放时刻。

`mode_events` 记录初始模式和每次请求的实际生效时间；`trace` 每段带有
生效模式。若错过截止，`MissRecord` 与 `explain_miss()` 同时保留
`release_mode`、`deadline_mode` 和按模式拆分的逐拍责任。截止拍属于
`[release, deadline)` 窗口之外；恰好在截止边界生效的切换会标记为
`deadline_mode`，但不会把新模式的任何一拍计入旧窗口的历史超期量。

## 保存与继续

`sys.snapshot()` / `System.restore(data)`（或 `save/load` 文件）
序列化完整运行态；测试验证断点续跑与一次跑完的执行记录逐段一致。

## 为功耗模型预留的扩展点

- `TaskSpec.attrs` 与 `Job.record` 是开放字典：将来挂功耗参数
  （频率/电压档、动态/静态功耗系数）和逐段能耗采样，**任务记录
  结构本身不需要重新设计**。
- 功耗按 trace 段结算即可：每段已知执行主体与时长。

## 布局

- `embsim/model.py` — 任务/中断/资源/开销的静态模型
- `embsim/system.py` — 虚拟处理器、调度器、继承、模式切换、快照
- `embsim/explain.py` — 错过截止时间的归因解释
- `tests/test_sim.py` — 手算核对的小例子（`python3 -m unittest discover -s tests`）
- `examples/demo.py` — 端到端演示
