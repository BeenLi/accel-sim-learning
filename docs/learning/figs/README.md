# 图源说明

本目录存放学习笔记用到的图，每张图在这里记一行来源。

- `official/`：从官方文档、白皮书、论文下载的图。
  - `cuda-grid-of-thread-blocks.png`：grid 由 thread block 组成，来源 [CUDA C++ Programming Guide · Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy)（原图 https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-thread-blocks.png ），2026-10-26 下载。用于 00-1。
  - `cuda-memory-hierarchy.png`：线程、block、cluster、grid 各自能访问的内存，来源 [CUDA C++ Programming Guide · Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-hierarchy)（原图 https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/memory-hierarchy.png ），2026-10-26 下载。用于 00-1。
- 仓库自带的图直接用相对路径引用，不复制：[`docs/img/`](../../img/)。（暂无引用）
- 教材（Aamodt 等，2018）有版权，书中的图不复制到这里。
- `drawn/`：官方没有、从源码读出后自己画的图，用 drawio-skill 手写 XML。每张图有 `.drawio`（可编辑源文件）和 `.png`（内嵌 drawio XML，可直接用 draw.io 打开编辑）。重新渲染：`drawio -x -f png -e -s 2 -o <name>.png <name>.drawio`，然后运行 drawio-skill 的 `scripts/repair_png.py <name>.png`。
  - `00-1-launch-to-model.png`：kernel 启动参数 → trace 头 → 模拟器字段的对应，以及 CTA 切成 warp 的规则。依据 [`trace_parser.cc:444-492`](../../../gpu-simulator/trace-parser/trace_parser.cc#L444-L492)、[`accel-sim.cc:174-192`](../../../gpu-simulator/accel-sim.cc#L174-L192)、[`inject_funcs.cu:78-92`](../../../util/tracer_nvbit/tracer_tool/inject_funcs.cu#L78-L92)、[`gpu-sim.cc:2099-2102`](../../../gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc#L2099-L2102)；例子数据来自本地 rodinia backprop 和 nn 的 trace。
