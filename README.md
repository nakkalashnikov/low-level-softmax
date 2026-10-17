# Hardware-Aware Softmax: Performance Engineering & Optimization

[![Language: C99](https://img.shields.io/badge/Language-C99-blue.svg)](https://en.wikipedia.org/wiki/C99)
[![Framework: PyTorch MPS](https://img.shields.io/badge/PyTorch-Apple_Silicon_MPS-EE4C2C.svg)](https://pytorch.org/)
[![Hardware: Apple Silicon](https://img.shields.io/badge/Target-Apple_Silicon_M1_Pro-black.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep-dive systems investigation into **hardware-aware optimization**, **memory bandwidth saturation**, and **kernel fusion** for the Softmax activation function across host CPU (ARM NEON) and GPU (Apple Metal / PyTorch MPS).

**Benchmark Dataset:** Single-precision floating-point tensor ($N = 10,000,000$ elements, $\approx 40\text{ MB}$).

---

## 📊 Benchmark Summary

Evaluated across four distinct execution paradigms to isolate the impacts of hardware acceleration, kernel fusion, memory access patterns, and SIMD vectorization:

| Implementation Paradigm | Platform / Backend | Execution Time | Memory Traffic | Primary Bottleneck / Characteristic |
|---|---|---|---|---|
| **PyTorch Official (`F.softmax`)** | Apple Silicon MPS | **6.20 ms** | $\approx 80\text{ MB}$ | Fused kernel; optimal shared memory utilization |
| **Pure C Optimized (3-Pass SIMD)** | Host CPU (M1 Pro) | **19.16 ms** | $\approx 200\text{ MB}$ | Memory bandwidth saturated ($\approx 8.45\text{ GB/s}$) |
| **Pure C Online Softmax (2-Pass)** | Host CPU (M1 Pro) | **35.22 ms** | $\approx 120\text{ MB}$ | Compute-bound; branch divergence breaks SIMD |
| **PyTorch Manual (Step-by-Step)** | Apple Silicon MPS | **94.99 ms** | $\approx 320\text{ MB}$ | Memory-bound; intermediate VRAM read/write roundtrips |

> **Key Systems Insight:**  
> Algorithmic complexity reduction does not guarantee faster hardware execution. On the host CPU, a theoretically "sub-optimal" 3-pass algorithm runs **1.83x faster** than the advanced 2-pass Online Softmax due to auto-vectorization mechanics and transcendental evaluation overhead.

---

## 🔍 Deep-Dive Technical Findings

### 1. PyTorch Architectural Gap: Fused vs. Step-by-Step (15.3x Speedup)
Executing manual tensor operations:
```python
max_val = torch.max(x)
safe_x = x - max_val
exps = torch.exp(safe_x)
out = exps / torch.sum(exps)
```
Forces the PyTorch MPS runtime to dispatch 5 discrete compute kernels sequentially. For a 40 MB input, this triggers catastrophic intermediate memory traffic ($\approx 320\text{ MB}$ total VRAM I/O), idling GPU compute cores.

In contrast, `torch.nn.functional.softmax` dispatches a **fused kernel**:
* Threadgroups load slices directly into ultra-fast on-chip Threadgroup Memory (Shared Memory).
* Max reduction, exponentiation, sum reduction, and division happen entirely within registers/L1 cache.
* Global VRAM traffic collapses to just **80 MB** (1 read, 1 write), achieving near-theoretical bandwidth efficiency.

---

### 2. Pure C CPU Optimization (19.16 ms on 10M Floats)
Located in [`softmax_cpu/softmax_cpu_effective.c`](softmax_cpu/softmax_cpu_effective.c), the C implementation leverages:

1. **Numerical Stability:** Subtracting $x_{\max}$ before exponentiation ($e^{x_i - x_{\max}}$) prevents IEEE-754 overflow.
2. **Strict Pointer Aliasing (`restrict`):** Applying `float *restrict` legally guarantees that input and output memory regions do not overlap. This allows Apple Clang (`-O3 -ffast-math`) to aggressively vectorize the loops with 128-bit **ARM NEON SIMD** registers (processing 4 floats per cycle).
3. **Reciprocal Multiplication:** Precomputing `1.0f / denominator_sum` once and multiplying the array reduces division operations from $10^7$ to $1$, saving dozens of clock cycles per element.
4. **Bandwidth Saturation:** Pumping $200\text{ MB}$ in $19.16\text{ ms}$ reaches $\approx 8.45\text{ GB/s}$, fully saturating a single performance core's allocation to the unified memory fabric.

---

### 3. The Algorithmic Paradox: Why Online Softmax Regressed on CPU
Online Softmax (Milakovsky et al. / FlashAttention foundation) dynamically rescales running accumulators:
$$S_{\text{new}} = S_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + 1.0$$
While this drops memory traffic from $200\text{ MB}$ to $120\text{ MB}$, wall-clock time on CPU **regressed by 83%** (35.22 ms vs 19.16 ms):
* **Branch Divergence:** The condition `if (input[i] > max_x)` inside the accumulator loop breaks compiler vectorization, forcing scalar execution.
* **Transcendental Tax:** `expf` is computed $2 \times 10^7$ times instead of $10^7$ times, starving the scalar ALU.
* *Conclusion:* Online Softmax is essential for massively parallel GPU threadgroups with fast shared memory, but streaming multi-pass SIMD wins on single-thread CPUs.

---

### 4. Custom Apple Metal GPU Kernel
Located in [`softmax_gpu/softmax_kernel.metal`](softmax_gpu/softmax_kernel.metal), the native Metal Shading Language implementation implements:
* Parallel reduction tree across 1024 threads per threadgroup.
* Inter-thread synchronization via `threadgroup_barrier(mem_flags::mem_threadgroup)`.
* Zero intermediate buffer allocations in global VRAM.

---

## 🛠️ Quickstart & Reproduction

### Prerequisites
* macOS with Apple Silicon (M-series chip)
* Clang / Xcode Command Line Tools
* Python 3.10+ with PyTorch (`pip install torch`)

### 1. Run Pure C SIMD Benchmark
```bash
clang -O3 -ffast-math -o softmax_cpu/softmax_cpu_effective softmax_cpu/softmax_cpu_effective.c
./softmax_cpu/softmax_cpu_effective
```

### 2. Run PyTorch MPS Comparison
```bash
python3 softmax.py
```

---

## 📜 Verification & Numerical Integrity
The maximum absolute difference between PyTorch MPS official output and the pure C implementation is bounded within machine epsilon ($\approx 4.55 \times 10^{-12}$), confirming absolute numerical fidelity across both execution paths.

---

## 📄 License
MIT License. Created by Nikita Kalashnikov.
