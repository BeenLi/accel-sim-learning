> 本目录是 Accel-Sim Framework 的学习文档，随仓库保存。文中的 `文件:行号` 都是相对本仓库的链接，点击可直接跳到对应代码行。

# Accel-Sim 学习计划：沿着"一条指令的一生"从 GPU 基础读到 Hopper 建模

> 学习对象是 Accel-Sim 2.0（trace 驱动前端 + NVBit tracer + 工具链）和它内嵌的 GPGPU-Sim 4.x 性能模型。路线先从零建立 GPU 编程模型和硬件组织的概念，再跑通一遍完整流程，然后顺着 kernel 下发、CTA 上 SM、warp 取指发射执行、访存经 L1/互连/L2/DRAM 的顺序逐层精读，最后读 PTX 模式、Hopper 新特性、统计与校准，并亲手改一次模型。最终目标是能改模型做体系结构研究。

---

## 0. 仓库速览（2026-10-26 勘察，只读）

勘察时的版本：Accel-Sim `c5e0872`；GPGPU-Sim 已原样固定进本仓库（上游 `accel-sim/gpgpu-sim_distribution` dev @ `91880c53`，本仓库提交 `b357625`）。行号以此为准。本仓库不跟踪上游，个人 GitHub 远端只作备份。

| 部分 | 位置 | 规模 | 备注 |
| --- | --- | --- | --- |
| Accel-Sim 主程序 | [`gpu-simulator/main.cc`](../../gpu-simulator/main.cc)、[`accel-sim.cc`](../../gpu-simulator/accel-sim.cc) | 约 350 行 | 主循环：读命令列表、按 stream 发射 kernel、推进时钟 |
| trace 解析 | [`gpu-simulator/trace-parser/`](../../gpu-simulator/trace-parser/) | 约 1.3k 行 | 解析 `kernelslist.g`、`.traceg`/`.tracez` |
| trace 驱动前端 | [`gpu-simulator/trace-driven/`](../../gpu-simulator/trace-driven/) | 约 1.4k 行 | 继承 GPGPU-Sim 的类，把"功能执行 PTX"换成"读 trace" |
| SASS opcode 表 | [`gpu-simulator/ISA_Def/`](../../gpu-simulator/ISA_Def/) | 约 1.8k 行 | 各代架构的 SASS opcode → 执行单元类别 |
| trace 配置 | [`gpu-simulator/configs/tested-cfgs/`](../../gpu-simulator/configs/tested-cfgs/) | 11 款 GPU | 每款一个 `trace.config` |
| GPGPU-Sim 时序模型 | [`gpu-simulator/gpgpu-sim/src/gpgpu-sim/`](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/) | 约 2.9 万行 | SM 流水线、cache、L2、DRAM、统计；固定在上游 `91880c53` |
| 互连网络 | [`gpu-simulator/gpgpu-sim/src/intersim2/`](../../gpu-simulator/gpgpu-sim/src/intersim2/) | 约 9.3k 行 | BookSim 改版的片上网络 |
| PTX 功能仿真 | [`src/cuda-sim/`](../../gpu-simulator/gpgpu-sim/src/cuda-sim/) + [`libcuda/`](../../gpu-simulator/gpgpu-sim/libcuda/) | 约 2.1 万 + 2.3 万行 | 假 libcudart、PTX 解析器、指令解释器；只在 PTX 模式用 |
| AccelWattch 功耗模型 | [`src/accelwattch/`](../../gpu-simulator/gpgpu-sim/src/accelwattch/) | 约 3 万行 | McPAT 改版 |
| GPU 配置 | [`gpu-simulator/gpgpu-sim/configs/tested-cfgs/`](../../gpu-simulator/gpgpu-sim/configs/tested-cfgs/) | 12 款 GPU | `gpgpusim.config` + 互连配置 + 功耗 xml |
| NVBit tracer | [`util/tracer_nvbit/`](../../util/tracer_nvbit/) | 约 2.4k 行 | 在真 GPU 上插桩抓 SASS trace |
| 跑仿真 / 收统计 | [`util/job_launching/`](../../util/job_launching/) | 约 4.5k 行 Python | `run_simulations.py`、`get_stats.py`，YAML 定义 app 和配置 |
| 画图 / correlation | [`util/plotting/`](../../util/plotting/)、[`util/hw_stats/`](../../util/hw_stats/) | 约 2.4k 行 Python | 仿真 vs 真机对比 |
| 调参 / 功耗脚本 | [`util/tuner/`](../../util/tuner/)、[`util/accelwattch/`](../../util/accelwattch/) | | |

- 构建：`source gpu-simulator/setup_environment.sh` 后用 CMake 构建，见原版 README [`README-accel-sim.md`](../../README-accel-sim.md) 的 Quick Start；本地已有 `gpu-simulator/bin/release/accel-sim.out`。
- 跑仿真：[`util/job_launching/run_simulations.py`](../../util/job_launching/run_simulations.py)。
- 抓 trace：[`util/tracer_nvbit/run_hw_trace.py`](../../util/tracer_nvbit/run_hw_trace.py)。
- 本地数据：`hw_run/` 下有 CUDA 11.0 的 rodinia_2.0-ft、ubench trace 和 QV100 真机统计（该目录被 `.gitignore` 忽略，链接只在本机有效）。
- 本地硬件：4 × NVIDIA L40（SM 8.9），CUDA 12.6 / 12.9 / 13.0。

## 1. 读者背景与目标

- 背景：做过体系结构研究，熟悉 CPU 微架构；GPU 体系结构和 GPGPU 教材没有读过，从头开始；C++ 熟练。
- 目标：
  - 理解 GPU 编程模型，以及一个 kernel 怎样被映射到硬件上，硬件内部的流水线和存储层次怎样工作；
  - 会用工具链跑实验（trace、仿真、统计、和真机对比）；
  - 理解 Hopper 新特性（TMA、WGMMA、mbarrier、cluster、chiplet）怎样建模；
  - 了解 PTX 执行驱动模式的实现；
  - 最终能改模型做研究。
- 参考书 / 资料：《General-Purpose Graphics Processor Architectures》（Aamodt、Fung、Rogers，2018，作者即 GPGPU-Sim / Accel-Sim 作者）；NVIDIA CUDA C++ Programming Guide；NVIDIA Volta / Hopper 架构白皮书。笔记会注明对应章节，但不依赖它们也能读懂。教材与阶段的对应：第 1–2 章（GPU 基础、编程模型、ISA）→ 00；第 3 章（SIMT core：SIMT 栈、warp 调度、operand collector）→ 04、05；第 4 章（L1/shared、互连、memory partition、L2、DRAM 调度）→ 06、07；§5.1（CTA 分配、调度、多 kernel）→ 03、04。Hopper（09）不在教材范围内，以 NVIDIA 白皮书和 Accel-Sim 2.0 论文为准。
- 标记：所有内容对读者都是新的，所以不使用 🆕，只用 ✅（从代码或配置直接读出）和 ❓（待核实）。
- 约束：长期投入，路线按细粒度拆。

## 2. 路线：第 00 步到第 11 步（2026-10-26 定）

先用一个概念阶段（00）建立 GPU 全貌，并亲手看一次真实的 PTX、SASS 和 trace；再跑通一遍工具链（01），让后面每读一个模块都能回到日志和统计里验证。02 到 07 顺着一条指令的生命周期往下读：骨架 → kernel 映射 → SM 前端 → SM 后端 → 片上存储 → 片外存储。08 读 PTX 模式时时序模型已经熟悉，只需关注"指令和数据从哪里来"。09 的 Hopper 特性都是对前面基线的改动，所以放在基线之后。10 学会量化，11 动手改模型，对准"做研究"的终点。

每篇精读都按"概念 → 代码怎么建模 → 用日志或统计验证"的顺序写。

| 阶段 | 主题 | 目标 | 总览 | 状态 |
| --- | --- | --- | --- | --- |
| 00 | GPU 基础：编程模型与硬件全貌 | 线程层次与内存空间、SIMT、硬件组织、PTX/SASS，并亲手看一次真实的 PTX/SASS/trace | [00 总览](00-0-gpu-basics-overview.md) | 进行中 |
| 01 | 跑通全流程与工具链 | trace 从哪来、怎么配置、怎么跑、结果在哪、怎么和真机对比 | [01 总览](01-0-toolchain-overview.md) | 框架 |
| 02 | 代码骨架与顶层数据流 | 主循环、trace 驱动的子类替换、时钟域、核心数据结构 | [02 总览](02-0-code-skeleton-overview.md) | 框架 |
| 03 | Kernel → 硬件映射 | kernel 选择、CTA 派发与 occupancy、warp 初始化、多 stream 并发 | [03 总览](03-0-kernel-mapping-overview.md) | 框架 |
| 04 | SM 流水线·前端 | 取指/译码、warp 调度、scoreboard、分支发散与 SIMT 栈、barrier | [04 总览](04-0-sm-frontend-overview.md) | 框架 |
| 05 | SM 流水线·后端 | operand collector 与寄存器 bank、执行单元、writeback | [05 总览](05-0-sm-backend-overview.md) | 框架 |
| 06 | 片上存储 | 访存合并、shared memory、ldst_unit、L1D、常量/纹理 cache | [06 总览](06-0-onchip-memory-overview.md) | 框架 |
| 07 | 片外存储 | 互连、L2 分区、地址映射、DRAM 时序与调度 | [07 总览](07-0-offchip-memory-overview.md) | 框架 |
| 08 | PTX 执行驱动模式 | 截获 CUDA 运行时、PTX 装载与解析、启动现场、功能解释器、SIMT 栈 | [08 总览](08-0-ptx-mode-overview.md) | 框架 |
| 09 | Hopper 新特性：概念与建模 | 异步搬运与 warp specialization、TMA、WGMMA、mbarrier、cluster/DSM、chiplet | [09 总览](09-0-hopper-overview.md) | 框架 |
| 10 | 统计、功耗与校准 | 计数器、AccelWattch、correlation 与 GPUVision、tuner | [10 总览](10-0-stats-power-calibration-overview.md) | 框架 |
| 11 | 综合实验 | 改一次模型，跑 benchmark，和基线、真机对比 | [11 总览](11-0-experiments-overview.md) | 框架 |

### 第 00 步：GPU 基础：编程模型与硬件全貌

目标：从零建立 GPU 的概念地图——程序员看到的线程和内存是什么、硬件怎样执行它们、硬件由哪些部件组成、代码怎样编译成 GPU 真正执行的指令——并在 00-5 用一个自己写的小 kernel 把这些概念和真实的 PTX、SASS、trace 一一对上。每个概念都落到模拟器里的一个类、一个字段或一个配置参数上。

- [x] 00-1 CUDA 编程模型：线程层次与内存空间（2026-10-26，结论在 [00-1 CUDA 编程模型](00-1-cuda-programming-model.md) 的"精读结论"）。读：本地 `hw_run/rodinia_2.0-ft/11.0/` 下各程序 `kernel-1.traceg` 的头部（grid、block、shmem、nregs、stream；`hw_run/` 不入库）、[`trace_parser.h:169`](../../gpu-simulator/trace-parser/trace_parser.h#L169) 的 `kernel_trace_t`、[`accel-sim.cc:174`](../../gpu-simulator/accel-sim.cc#L174) 的 `create_kernel_info`、[`abstract_hardware_model.h:522`](../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h#L522) 的 `kernel_info_t`。问题：启动一个 kernel 时程序员指定了什么（grid、block、shared memory、stream）？线程怎样编号并分组成 block 和 warp？global / shared / local / constant / param 各个内存空间存在哪、谁能看见、生命周期多长？这些信息在 trace 头和 `kernel_info_t` 里对应哪个字段？
- [ ] 00-2 SIMT 执行模型：warp、活跃掩码与延迟隐藏。读：同一批 trace 指令行的 mask 列（格式说明在每个 `kernel-N.traceg` 头部之后的 `#traces format` 行）、[`shader.h:111`](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h#L111) 的 `shd_warp_t`、[`abstract_hardware_model.h:747`](../../gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h#L747) 的 `simt_stack`、[`shader.h:748`](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h#L748) 的 `scheduler_unit`。问题：为什么 32 个线程组成一个 warp 一起执行？活跃掩码、分支发散、重汇聚是怎么回事？GPU 怎样靠在大量 warp 之间切换来隐藏延迟，和 CPU 的乱序执行、SMT 相比取舍在哪？这些概念在模拟器里分别对应哪个类？
- [ ] 00-3 GPU 硬件组织：从 GPC 到 DRAM，用 QV100 和 H100 的真实数字。读：[`SM7_QV100/gpgpusim.config:64-83`](../../gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim.config#L64-L83)（SM 数、显存通道、时钟、寄存器、线程上限）、[`:114-165`](../../gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM7_QV100/gpgpusim.config#L114-L165)（sub-core、调度器、L1/shared、L2）、[`SM90_H100/gpgpusim.config:24-120`](../../gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config#L24-L120)；模拟器的层次类 [`gpu-sim.h:604`](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.h#L604) `gpgpu_sim`、[`shader.h:3292`](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h#L3292) `simt_core_cluster`、[`shader.h:2713`](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h#L2713) `shader_core_ctx`、[`l2cache.h:239`](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.h#L239) `memory_partition_unit`、[`l2cache.h:331`](../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.h#L331) `memory_sub_partition`。问题：GPC/TPC/SM/sub-core、寄存器堆、shared memory/L1、L2 切片、显存通道怎样层层组织？QV100 和 H100 各有多少 SM、多少寄存器、多大 L1/shared/L2、多少显存通道、各时钟域多少 MHz？模拟器用哪几个类表示这些层次，配置参数怎样对应到它们？
- [ ] 00-4 编译流程与 ISA：CUDA C++ → PTX → SASS。读：[`volta_opcode.h:12-24`](../../gpu-simulator/ISA_Def/volta_opcode.h#L12-L24)（binary version 与 opcode 表）、[`trace_opcode.h:11`](../../gpu-simulator/ISA_Def/trace_opcode.h#L11)（统一的 opcode 枚举）、[`trace_driven.cc:199-212`](../../gpu-simulator/trace-driven/trace_driven.cc#L199-L212)（按 binary version 选 opcode 表）、trace 头里的 `binary version = 70`。问题：nvcc 怎样把 CUDA C++ 编译成 PTX（虚拟 ISA），再由 ptxas 生成 SASS（真实 ISA），fatbinary 里装了什么？PTX 和 SASS 的关键区别是什么（虚拟寄存器 vs 物理寄存器、跨代稳定 vs 每代不同）？Accel-Sim 为什么选择 SASS trace？一条 SASS opcode 怎样被归到执行单元类别（`SP_OP` 等）？
- [ ] 00-5 动手：从一个小 kernel 到 PTX、SASS 和 trace。读：[`util/tracer_nvbit/README.md`](../../util/tracer_nvbit/README.md)、[`tracer_tool.cu:282`](../../util/tracer_nvbit/tracer_tool/tracer_tool.cu#L282) `nvbit_at_init`、[`:377`](../../util/tracer_nvbit/tracer_tool/tracer_tool.cu#L377) `instrument_function_if_needed`、[`:856`](../../util/tracer_nvbit/tracer_tool/tracer_tool.cu#L856) `nvbit_at_cuda_event`、[`inject_funcs.cu:40`](../../util/tracer_nvbit/tracer_tool/inject_funcs.cu#L40) `instrument_inst`、[`post-traces-processing.cpp:172`](../../util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing.cpp#L172)、[`torch_hook/run.sh`](../../util/tracer_nvbit/others/torch_hook/run.sh)。动手：写一个包含 vector add、`if` 分支和 shared memory 归约的小 kernel，用 nvcc 编译，`cuobjdump -ptx` / `-sass` 看两层代码，在本机 L40 上用 NVBit tracer 抓 trace。问题：同一段源码在 PTX、SASS、trace 三层分别长什么样？`threadIdx`/`blockIdx`、分支、`__syncthreads`、shared memory、global load/store 各对应哪些指令？trace 的每一列（PC、mask、寄存器、地址）怎样对应 00-1 到 00-4 的概念？L40 是 SM 8.9，而 [`trace_driven.cc:199-212`](../../gpu-simulator/trace-driven/trace_driven.cc#L199-L212) 没有 89 的 opcode 表，编译时选哪个 `-arch` 才能让 trace 被模拟器接受 ❓？

### 第 01 步：跑通全流程与工具链

目标：用本地已有的 trace 在 QV100 配置上跑通一次仿真，看懂输入、配置、输出和 correlation。（开始这一步时再细化成逐篇待办）

- [ ] trace 文件格式与 tracer 流程：`kernelslist.g`、`.traceg` / `.tracez`、`run_hw_trace.py`、post-processing
- [ ] 配置系统：`option_parser`、`gpgpusim.config` 与 `trace.config` 的分工、`define-standard-cfgs.yml`
- [ ] 跑一次仿真：`run_simulations.py`、运行目录结构、`justrun.sh`、输出日志的结构
- [ ] 收集统计与 correlation：`get_stats.py`、`util/plotting/`、和 QV100 真机数据对比
- [ ] 动手：给本仓库写自己的 CI（格式检查 → 在上游 Docker 镜像里构建 → QV100 SASS 冒烟测试），替换依赖 Purdue 自托管 runner 的上游工作流，完成后重新打开仓库的 Actions。2026-10-26 已核实的前提：
  - 格式检查：`pip install clang-format==16.0.6` 后运行三个 `format-code.sh`（accel-sim、tracer、gpgpu-sim），当前代码没有任何改动，可以直接通过。
  - 镜像 `ghcr.io/accel-sim/accel-sim-framework:ubuntu-24.04-cuda-12.8-minimal` 可匿名拉取；本机有 docker，可以先在同一镜像里本地试跑。
  - trace：`https://engineering.purdue.edu/tgrogers/accel-sim/traces/tesla-v100/latest/rodinia_2.0-ft.tgz`（21 MB，解压到 `hw_run/rodinia_2.0-ft/9.1`），做法参考 [`.github/scripts/main-docker.sh`](../../.github/scripts/main-docker.sh)。
  - 没有 Slurm 时 `run_simulations.py` 自动改用本地 `procman.py`。
  - 触发：代码改动的 push（`paths-ignore: docs/**, '**.md'`，提交笔记不触发）+ 手动触发。不做 tracer job（要 GPU，公开仓库不挂自托管 runner）。
  - 待定：GitHub runner 只有 4 核、单 job 6 小时上限，先在本地给 rodinia 计时，必要时删减 app。

### 第 02 步：代码骨架与顶层数据流

目标：看懂从 `main` 到每个时钟周期的调用链，以及 trace 驱动是怎样"插"进 GPGPU-Sim 的。

- [ ] Accel-Sim 主循环：命令列表、kernel 窗口、stream 忙闲、`simulate()` / `cleanup()`
- [ ] trace 驱动的子类替换：`trace_gpgpu_sim` / `trace_simt_core_cluster` / `trace_shader_core_ctx` / `trace_shd_warp_t` / `trace_warp_inst_t`，以及和 PTX 模式的分界（`func_exec_inst`）
- [ ] `gpgpu_sim::cycle()` 与时钟域：core / icnt / L2 / DRAM 四个时钟域的推进顺序
- [ ] 核心数据结构：`warp_inst_t`、`kernel_info_t`、`mem_fetch`、`gpgpu_context`

### 第 03 步：Kernel → 硬件映射

目标：看清一个 kernel 的 CTA 怎样被分到 SM 上、warp 怎样被创建并绑定 trace。

- [ ] kernel 启动与选择：`gpgpu_sim::launch`、`select_kernel`、并发 kernel
- [ ] CTA 派发与 occupancy：`issue_block2core`、寄存器 / shared memory / warp 槽 / CTA 槽的资源检查
- [ ] warp 初始化与 trace 绑定：`init_warps`、`init_traces`、`get_next_threadblock_traces`

### 第 04 步：SM 流水线·前端

目标：看懂一条指令从取指到发射之间经历的每一级和每一个阻塞原因。

- [ ] 流水线总览：`shader_core_ctx::cycle()` 的倒序调用与流水寄存器
- [ ] 取指与译码：I-cache、`ifetch_buffer_t`、ibuffer
- [ ] warp 调度器：LRR / GTO / two-level 等策略、sub-core 划分
- [ ] scoreboard 与发射条件
- [ ] 分支发散、SIMT 栈（trace 模式下怎样处理）与 barrier

### 第 05 步：SM 流水线·后端

目标：看懂发射之后指令怎样读操作数、在哪个执行单元执行多久、怎样写回。

- [ ] operand collector 与寄存器 bank 冲突
- [ ] 执行单元：SP / DP / SFU / INT / Tensor / specialized 的延迟与发射间隔，以及 `trace.config` 里的延迟配置
- [ ] writeback 与 scoreboard 释放

### 第 06 步：片上存储

目标：看懂访存指令在 SM 内部怎样被拆成请求、怎样访问 shared memory 和 L1。

- [ ] 访存合并（coalescing）：一条 warp 访存怎样生成若干个 `mem_fetch`
- [ ] shared memory 与 bank 冲突
- [ ] `ldst_unit` 的结构与流水
- [ ] sectored L1D：tag、MSHR、写策略、L1/shared 容量划分
- [ ] 常量 cache 与纹理 cache

### 第 07 步：片外存储

目标：看懂请求离开 SM 之后，经互连到 L2、再到 DRAM、再返回的完整路径。

- [ ] 互连：`icnt_wrapper`、`local_interconnect` 与 `intersim2` 的选择
- [ ] memory partition 与 L2 sub-partition
- [ ] 地址映射与 hashing：`addrdec`、`hashing`
- [ ] DRAM 时序模型与 FR-FCFS 调度

### 第 08 步：PTX 执行驱动模式

目标：看懂不靠 trace、而是功能模拟 PTX 时，指令和数据从哪里来，以及它和时序模型怎样衔接。

- [ ] 截获 CUDA 运行时：假的 `libcudart.so`、`LD_LIBRARY_PATH`、显存分配与 memcpy
- [ ] 装载：`__cudaRegisterFatBinary` → cuobjdump → PTX 词法 / 语法分析 → IR 与符号表
- [ ] 启动现场：param 空间、特殊寄存器、各内存空间的稀疏实现
- [ ] 功能解释器：`instructions.cc`，以及功能执行与时序模型的衔接
- [ ] SIMT 栈与后支配重汇聚、PTXPlus、PTX 模式与 trace 模式的精度对比

### 第 09 步：Hopper 新特性：概念与建模

目标：理解 Hopper 新增的异步、分布式机制，以及 Accel-Sim 2.0 怎样在基线模型上建模它们。

- [ ] 概念：异步数据搬运与 warp specialization（以 FlashAttention-3 为例）
- [ ] TMA：trace 里的 TMA 地址信息与数据通路
- [ ] WGMMA：异步矩阵乘
- [ ] mbarrier：producer / consumer 同步
- [ ] threadblock cluster 与 DSM（分布式 shared memory）
- [ ] chiplet / uGPU 分区内存与 LRC

### 第 10 步：统计、功耗与校准

目标：会定义和读统计、会算功耗、会做 correlation 和调参。

- [ ] 统计计数器：定义、更新、输出
- [ ] AccelWattch：从事件计数到功耗
- [ ] correlation 流程与 GPUVision
- [ ] tuner：用 microbenchmark 调配置

### 第 11 步：综合实验

目标：独立完成一次"改模型 → 跑实验 → 对比分析"。

- [ ] 改动设计与实现（例如新调度策略或新计数器）
- [ ] 跑 rodinia / ubench，和基线、QV100 真机数据对比并分析

---

## 3. 索引：资料在哪

| 内容 | 位置 |
| --- | --- |
| 笔记的写法 | [repo-learning-path](https://github.com/BeenLi/arch-reading-skills/tree/main/skills/repo-learning-path) skill（[arch-reading-skills](https://github.com/BeenLi/arch-reading-skills)） |
| 学习笔记：00 GPU 基础 | [00 总览](00-0-gpu-basics-overview.md)（进行中） |
| 图源说明 | [`figs/README.md`](figs/README.md) |
| 仓库自带文档 | [`README-accel-sim.md`](../../README-accel-sim.md)（原版 README）、[`release.notes.md`](../../release.notes.md)、[`gpgpu-sim4.md`](../../gpu-simulator/gpgpu-sim4.md)、[`AccelWattch.md`](../../AccelWattch.md) |
| 各工具说明 | [`util/tracer_nvbit/README.md`](../../util/tracer_nvbit/README.md)、[`util/job_launching/README.md`](../../util/job_launching/README.md)、[`util/plotting/README.md`](../../util/plotting/README.md)、[`util/tuner/README.md`](../../util/tuner/README.md) |
| 论文 | Accel-Sim 2.0（[arXiv 2608.22602](https://arxiv.org/abs/2608.22602)）、Accel-Sim ISCA'20（[PDF](https://people.ece.ubc.ca/~aamodt/publications/papers/accelsim.isca2020.pdf)）、AccelWattch MICRO'21（[PDF](http://paragon.cs.northwestern.edu/papers/2021-MICRO-AccelWattch-Kandiah.pdf)） |
| 教材 | Aamodt, Fung, Rogers,《General-Purpose Graphics Processor Architectures》, Morgan & Claypool, 2018 |
| 官方文档 | [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)、[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)、[CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/) |
