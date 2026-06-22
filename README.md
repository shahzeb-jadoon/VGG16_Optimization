# VGG-16 CUDA Optimization

**High Performance Architectures**

Custom CUDA C++ inference for VGG-16 on **NVIDIA GeForce RTX GPU** (CC 8.6, 16 GB VRAM), using cuDNN (conv), cuBLAS (FC), and hand-written kernels (shared-memory tiling, memory coalescing, kernel fusion). Profiled with **NVIDIA Nsight Compute**.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-shahzebkjadoon-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/shahzebkjadoon)
[![CUDA](https://img.shields.io/badge/CUDA-C%2B%2B-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

---

## Results (FC layer · batch = 16)

The fully-connected layer is implemented as a dense GEMM (`fc_matmul_tiled_kernel`). Roofline / CGMA analysis on the **project GPU** (RTX GPU) shows the kernel is **not compute-bound**: arithmetic intensity is far below the ridge point needed to saturate FP32 peak.

| Kernel | GFLOPS @ batch=16 | CGMA (FLOP/byte) | Ridge (compute-bound) | Bottleneck |
|---|---:|---:|---:|---|
| `fc_matmul_tiled_kernel` | ~41.5 | ~2.0 | ~156.7 | Memory-latency / low AI (~8% of FP32 peak) |

**Scaling:** near-linear FC throughput scaling from batch=1 → batch=16 (~16.4×), measured on the project machine.

> **Note:** Nsight SOL shows compute % ≈ memory % (~68% at batch≈1, ~88% at batch=16) and labels the kernel "balanced" — that is different from roofline classification. Low CGMA means the matmul sits on the **memory roof**, not the compute ceiling.

---

## Profiling (Nsight Compute)

Captures from `fc_matmul_tiled_kernel` on RTX GPU Laptop GPU. Full `.ncu-rep` files are kept locally under `profile/` (gitignored).

### GPU Speed of Light + workload analysis

Compute (SM) vs memory throughput, L1/L2/DRAM breakdown, and roofline note (~8% of FP32 peak).

### Memory access patterns

Global **loads** fully coalesced (32/32 bytes per sector); global **stores** partially uncoalesced (~8/32 bytes) — top optimization lever (~51% estimated speedup in Nsight).

### Launch statistics & occupancy

37 registers/thread, ~72.8% achieved occupancy (register-limited vs 100% theoretical), grid `(1, 256, 1)` at batch≈1.

### Optional — source counters & workload balance

Branch efficiency 100%, 0 divergent branches; mild SM workload imbalance (~9%).

---

## What we built

- **Custom CUDA kernels** for VGG-16 conv & FC layers (tiled + vectorized variants)
- **cuDNN** convolutions and **cuBLAS** FC baselines for comparison
- **ONNX Runtime** inference path (`vgg16.onnx`)
- **Nsight Compute** profiling (`ncu`) with batch-size sweep (1 → 16)
- **CMake + Ninja** build on Windows (Visual Studio 2022 toolchain)

---

## Quick start

### Prerequisites

- **CUDA Toolkit** 12.x (tested with RTX GPU, CC 8.6)
- **cuDNN** (system install or `external/cudnn/`)
- **CMake** 3.18+, **Ninja**, **Visual Studio 2022** (Desktop C++)
- **ONNX Runtime** (fetched by CMake if not system-wide)

### Model (not in repo)

The VGG-16 ONNX model is **not committed** (~500 MB). Export or place locally:

```text
models/vgg16.onnx
