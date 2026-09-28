# 嵌入式任务仿真系统（第一版）

单核虚拟处理器上的可抢占实时调度仿真，纯 Python、零依赖。

## 模型

- **Task**：固定优先级周期任务，作业由若干 Section 组成；段可声明所需共享资源。Task.attrs / Section.attrs 是扩展位——将来加功耗模型（能耗/电压档位）直接挂在 attrs 上，任务记录结构不变，快照序列化自动覆盖。
- **Interrupt**：指定时刻触发的 ISR，抢占一切任务，ISR 之间按优先级抢占，耗时计入时间轴。
- **Resource**：互斥资源，FIFO 等待队列。

## 优先级反转规则：基本优先级继承（可传递）

持有资源的任务被更高优先级任务阻塞时，临时继承所有等待者中最高的有效优先级，释放资源后恢复；继承沿阻塞链传递。本版未实现优先级天花板（后续版本候选）。

## 时间记账

抢占、阻塞、ISR、（可选的）切换开销都占用时间轴；trace 是首尾相接、无缝隙无重叠的逐段执行记录 Segment(start, end, who, kind, reason)，每段注明结束原因（段完成/被中断抢占/被新释放任务抢占/作业完成）。

## 错过截止时间解释

Simulator.missed_deadlines() 把出错作业的 [release, finish] 窗口分解为：自身执行 / 被高优先级抢占 / 低优先级持资源反转 / 中断占用 四类，给出定量解释。

## 保存与续跑

snapshot() 生成 JSON 字节串，Simulator.restore() 恢复后继续 run()，结果与不中断连续运行一致（测试覆盖）。

## 文件

- rtos_sim.py — 仿真器
- test_rtos_sim.py — 手算场景核对（含时间轴注释）、继承、错过解释、快照续跑
- demo.py — 演示：t=5 快照，恢复后续跑，打印 trace 与错过分析

运行：python3 -m unittest test_rtos_sim -v ； python3 demo.py
