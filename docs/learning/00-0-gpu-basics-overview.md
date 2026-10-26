> 本文是 Accel-Sim Framework 的学习文档，随仓库保存；图在 `figs/`，来源见 [`figs/README.md`](figs/README.md)。文中的 `文件:行号` 都是相对本仓库的链接，点击可直接跳到对应代码行。

# 00 GPU 基础：编程模型与硬件全貌总览

> 学习计划第 00 步，见 [学习计划](README.md)。标 ✅ 的是直接从代码或配置读出的事实，标 ❓ 的是待核实。框架已搭，内容待写。

---

## 0. 一张图：GPU 编程模型、SIMT 执行与硬件层次的概念地图

（待写）

---

## 1. 细节分篇

| 篇 | 内容 | 主要结论 |
| --- | --- | --- |
| 00-1 CUDA 编程模型：线程层次与内存空间（待写） | grid / block / warp / thread 与各内存空间，及其在 trace 头和 `kernel_info_t` 中的对应 | |
| 00-2 SIMT 执行模型：warp、活跃掩码与延迟隐藏（待写） | warp 锁步执行、分支发散与重汇聚、靠 warp 切换隐藏延迟（对比 CPU） | |
| 00-3 GPU 硬件组织：从 GPC 到 DRAM（待写） | GPC/SM/sub-core、寄存器堆、L1/shared、L2、显存通道；QV100 与 H100 的真实参数 | |
| 00-4 编译流程与 ISA：CUDA C++ → PTX → SASS（待写） | nvcc/ptxas/fatbinary、PTX 与 SASS 的区别、SASS opcode 到执行单元的映射 | |
| 00-5 动手：从一个小 kernel 到 PTX、SASS 和 trace（待写） | 自己写 kernel，逐层对照源码、PTX、SASS、NVBit trace | |

---

## 2. 待核实清单

（待写）
