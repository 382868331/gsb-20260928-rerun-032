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
并输出中文报告。演示：`python3 examples/demo.py`。

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
- `embsim/system.py` — 虚拟处理器、调度器、继承、快照
- `embsim/explain.py` — 错过截止时间的归因解释
- `tests/test_sim.py` — 手算核对的小例子（`python3 -m unittest discover -s tests`）
- `examples/demo.py` — 端到端演示
