# Accel-Sim 学习笔记

这是一个**个人学习仓库**：把 [Accel-Sim 2.0](https://github.com/accel-sim/accel-sim-framework)（周期级 GPU 模拟器）固定在一个版本上，从 GPU 基础概念开始，逐模块精读源码，并把笔记公开在 [`docs/learning/`](docs/learning/README.md)。

- 这不是 Accel-Sim 官方仓库，不跟踪、也不向上游提交改动。要用模拟器做正式工作，请到 [accel-sim/accel-sim-framework](https://github.com/accel-sim/accel-sim-framework)。
- 原版 README（安装依赖、Quick Start、Hopper 支持、vLLM 追踪、引用方式）保留在 [`README-accel-sim.md`](README-accel-sim.md)。

## 固定的版本

| 组件 | 上游仓库 | 固定的版本 |
| --- | --- | --- |
| Accel-Sim Framework | [accel-sim/accel-sim-framework](https://github.com/accel-sim/accel-sim-framework) | `dev` @ `c5e0872` |
| GPGPU-Sim（性能模型） | [accel-sim/gpgpu-sim_distribution](https://github.com/accel-sim/gpgpu-sim_distribution) | `dev` @ `91880c53`，原样放入 [`gpu-simulator/gpgpu-sim/`](gpu-simulator/gpgpu-sim/)，不再单独 clone |

笔记里的 `文件:行号` 链接都以这个版本为准。

## 学习笔记

从 [学习计划](docs/learning/README.md) 开始读。路线沿着"一条指令的一生"展开：

| 阶段 | 主题 |
| --- | --- |
| 00 | [GPU 基础：编程模型与硬件全貌](docs/learning/00-0-gpu-basics-overview.md) |
| 01 | [跑通全流程与工具链](docs/learning/01-0-toolchain-overview.md) |
| 02 | [代码骨架与顶层数据流](docs/learning/02-0-code-skeleton-overview.md) |
| 03 | [Kernel → 硬件映射](docs/learning/03-0-kernel-mapping-overview.md) |
| 04 | [SM 流水线·前端](docs/learning/04-0-sm-frontend-overview.md) |
| 05 | [SM 流水线·后端](docs/learning/05-0-sm-backend-overview.md) |
| 06 | [片上存储](docs/learning/06-0-onchip-memory-overview.md) |
| 07 | [片外存储](docs/learning/07-0-offchip-memory-overview.md) |
| 08 | [PTX 执行驱动模式](docs/learning/08-0-ptx-mode-overview.md) |
| 09 | [Hopper 新特性：概念与建模](docs/learning/09-0-hopper-overview.md) |
| 10 | [统计、功耗与校准](docs/learning/10-0-stats-power-calibration-overview.md) |
| 11 | [综合实验](docs/learning/11-0-experiments-overview.md) |

每篇笔记按"概念 → 代码怎么建模 → 用日志或统计验证"展开。标 ✅ 的是直接从代码或配置读出的事实，标 ❓ 的是待核实。进度见学习计划里的勾选框。

## 仓库目录

```
.
├── README.md                  本文件：仓库说明与目录
├── README-accel-sim.md        Accel-Sim 原版 README
├── docs/
│   ├── learning/              学习笔记（学习计划、各阶段总览与精读、figs/ 图）
│   └── img/                   原版 README 用到的图
├── gpu-simulator/             模拟器本体
│   ├── main.cc, accel-sim.cc  入口与主循环：读命令列表、发射 kernel、推进时钟
│   ├── trace-parser/          解析 kernelslist.g 与 .traceg / .tracez trace
│   ├── trace-driven/          trace 驱动前端：继承 GPGPU-Sim 的类，用 trace 代替功能仿真
│   ├── ISA_Def/               各代架构的 SASS opcode 表
│   ├── configs/tested-cfgs/   各款 GPU 的 trace.config
│   ├── gpgpu-sim/             GPGPU-Sim 性能模型（固定版本）
│   │   ├── src/gpgpu-sim/     时序模型：SM 流水线、cache、L2、DRAM、统计
│   │   ├── src/intersim2/     片上互连网络
│   │   ├── src/cuda-sim/      PTX 功能仿真（PTX 模式）
│   │   ├── libcuda/           模拟用的 CUDA runtime（PTX 模式）
│   │   ├── src/accelwattch/   AccelWattch 功耗模型
│   │   └── configs/tested-cfgs/  各款 GPU 的 gpgpusim.config
│   └── setup_environment.sh   构建前要 source 的环境脚本
├── util/
│   ├── tracer_nvbit/          NVBit tracer：在真 GPU 上抓 SASS trace
│   ├── job_launching/         批量跑仿真、监控、收集统计
│   ├── plotting/, hw_stats/   仿真与真机的 correlation 与画图
│   ├── tuner/                 用 microbenchmark 调配置
│   └── accelwattch/           功耗模型的批处理与标定脚本
├── short-tests.sh             上游的冒烟测试脚本
├── release.notes.md           Accel-Sim 2.0 发布说明
├── AccelWattch.md, mlperf.md  上游的专题文档
└── LICENSE, CITATION.*        许可证与引用信息
```

`hw_run/`（trace 与真机数据）和 `sim_run_*/`（仿真运行目录）是本地生成的，不入库。

## 构建与运行

和上游一致，详见 [`README-accel-sim.md`](README-accel-sim.md) 的 Quick Start：

```bash
source ./gpu-simulator/setup_environment.sh      # 使用仓库内的 gpgpu-sim，不会再去 clone
cmake -S ./gpu-simulator/ -B ./gpu-simulator/build
cmake --build ./gpu-simulator/build -j8
cmake --install ./gpu-simulator/build            # 得到 ./gpu-simulator/bin/release/accel-sim.out
```

## 许可与引用

代码版权归 Accel-Sim / GPGPU-Sim 原作者，按 [BSD 2-Clause 许可证](LICENSE) 发布（GPGPU-Sim 部分见 [`gpu-simulator/gpgpu-sim/COPYRIGHT`](gpu-simulator/gpgpu-sim/COPYRIGHT)）。学术使用请按 [`README-accel-sim.md`](README-accel-sim.md#how-to-cite) 的 "How to Cite" 引用原论文。`.github/` 下的 CI 工作流是上游的，本仓库不使用。
