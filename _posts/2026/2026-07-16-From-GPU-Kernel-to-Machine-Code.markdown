---
layout: post
title: "[ML] From GPU Kernel to Machine Code"
date: 2026-07-16 13:19
subtitle: Triton, CUDA, PTX, and SASS
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - ML
---
## Overview

When you write a GPU operation, several different languages and compilation
stages may sit between your source code and the instructions executed by the
GPU.

The short version is:

```
CUDA C++ or Triton
        │
        ▼
compiler intermediate representations
        │
        ▼
PTX virtual GPU instructions
        │
        ▼
GPU-specific binary and native instructions
        │
        ▼
NVIDIA GPU
```

This article explains what each layer does, where Triton can replace handwritten
CUDA C++, and where it cannot.

## 1. What CUDA means

**CUDA** is NVIDIA's complete GPU-computing platform. It includes:

- the CUDA C++ programming model;
- **compiler tools** such as `nvcc` and `ptxas`;
- the **CUDA runtime** and driver APIs;
- **optimized libraries** such as **cuBLAS and cuFFT**;
- **debugging** and **profiling tools**;
- support for **memory management**, **streams, events, graphs, and synchronization**.

This matters because the question "Can Triton replace CUDA?" is too broad.
Triton can replace **handwritten CUDA C++ for some kernels**, but it does not
replace the complete CUDA platform underneath those kernels.

## 2. What a GPU kernel is

A **kernel** is a function executed in parallel on the GPU.

For example, adding two arrays can be expressed conceptually as:

```c
for every element i in parallel:
    output[i] = x[i] + y[i]
```

In CUDA C++, the programmer explicitly works with threads, blocks, and thread
indices:

```cpp
__global__ void add_kernel(
    const float* x,
    const float* y,
    float* output,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        output[i] = x[i] + y[i];
    }
}
```

The launch configuration specifies how many blocks and threads to create:

```cpp
add_kernel<<<num_blocks, threads_per_block>>>(x, y, output, n);
```

CUDA C++ gives the programmer detailed control, but that also means the
programmer must reason about:

- thread and block organization;
- memory coalescing;
- shared memory;
- synchronization;
- register usage;
- occupancy;
- hardware-specific behavior.

## 3. What Triton is

**Triton is a Python-based language and compiler for writing GPU kernels.**
Instead of assigning individual elements to CUDA threads directly, Triton
usually expresses work over blocks of tensor elements.

A simplified Triton addition kernel looks like:

```python
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
```

The programmer still chooses important algorithmic properties - such as "how work
is divided into blocks and how data is loaded”, but the compiler handles much of the mapping
onto GPU threads.

Triton is particularly useful for:

- fused elementwise operations;
- reductions and normalization;
- softmax and attention;
- matrix multiplication;
- custom tensor preprocessing and postprocessing.

The official tutorials demonstrate examples such as fused softmax, matrix
multiplication, layer normalization, and attention:
[Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html).

## 4. Triton versus CUDA C++

Triton and CUDA C++ overlap at the **kernel-authoring level**, but their scope
and tradeoffs differ.

| Question                      | Triton                         | CUDA C++                                 |
| ----------------------------- | ------------------------------ | ---------------------------------------- |
| Programming model             | Blocks of tensor elements      | Threads and thread blocks                |
| Language                      | Python-based DSL               | C++ extensions                           |
| Boilerplate                   | Usually lower                  | Usually higher                           |
| Low-level control             | Moderate                       | Maximum                                  |
| Best fit                      | Dense, tensor-oriented kernels | General and irregular GPU algorithms     |
| PyTorch compiler integration  | Strong                         | Usually custom extension or library call |
| Access to new NVIDIA features | May lag or abstract them       | Usually available more directly          |
| Ecosystem maturity            | Growing quickly                | Large and mature                         |

### Advantages of Triton

- **Higher-level programming model:** It removes much thread-level boilerplate.
- **Fast experimentation:** Kernel layouts and block sizes are easy to vary.
- **Natural fusion:** Multiple tensor operations can be combined into one
  kernel.
- **Good PyTorch integration:** TorchInductor frequently generates Triton
  kernels automatically.
- **Readable numerical code:** A kernel can resemble the blocked tensor
  algorithm more closely than its CUDA thread mapping.

### Advantages of CUDA C++

- **Maximum control:** Threads, memory, synchronization, streams, and
  hardware-specific behavior are directly accessible.
- **Broader scope:** CUDA can express irregular algorithms and complete GPU
  applications, not only tensor-style kernels.
- **Mature tooling and libraries:** CUDA has extensive profilers, debuggers,
  documentation, and production history.
- **A stronger escape hatch:** Algorithms that do not fit Triton's block model
  can still be expressed directly.
- **Earlier hardware access:** New NVIDIA capabilities may be available through
  CUDA before Triton exposes a convenient abstraction.

### Which one is faster?

Neither is automatically faster.

A handwritten Triton kernel may lose to a highly tuned CUDA library. Conversely,
a fused Triton kernel may beat a sequence of efficient CUDA kernels because it
avoids:

- intermediate memory writes;
- intermediate memory reads;
- additional kernel launches.

The correct comparison uses realistic:

- tensor shapes;
- data types;
- input layouts;
- batch sizes;
- surrounding operations.

---

## 5. What PTX is

**PTX** means **Parallel Thread Execution**. It is NVIDIA's **virtual instruction**
**set** for GPU programs.

PTX is lower-level than CUDA C++ or Triton, but it is generally not the native
instruction encoding executed by one specific GPU.

A small PTX fragment may resemble:

```ptx
ld.global.f32  %f1, [%rd1];
ld.global.f32  %f2, [%rd2];
add.f32        %f3, %f1, %f2;
st.global.f32  [%rd3], %f3;
```

PTX abstracts over physical GPU instruction sets. A compiler or NVIDIA driver
can translate it into code for a particular GPU architecture.

NVIDIA defines PTX as a virtual machine and instruction-set architecture in the
[PTX ISA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

## 6. `compute_86` versus `sm_86`

CUDA uses two related target labels:

```text
compute_86    PTX virtual architecture target
sm_86         Real GPU architecture target
```

For example:

- compiling for `compute_86` produces PTX that assumes the capabilities of that
  virtual architecture;
- compiling for `sm_86` produces executable code for GPUs with that streaming
  multiprocessor architecture.

The exact compatibility rules depend on what code is packaged and which GPU
features it uses.

## 7. PTX, cubin, fatbin, and SASS

These terms describe different things:

- PTX: Virtual GPU instruction set.
- SASS: Common name for the **native GPU instructions** executed by **a particular NVIDIA architecture**.
- cubin: ELF-format binary containing GPU code for a particular architecture.
- fatbin: Package that can contain code for multiple architectures, including several cubins and/or PTX.

A simplified compilation chain is:

```
CUDA C++ or Triton source
            │
            ▼
      compiler IR stages
            │
            ▼
        PTX virtual ISA
            │
        ┌───┴───────────────┐
        ▼                   ▼
   ptxas offline       NVIDIA driver JIT
   compilation         compilation
        │                   │
        └─────────┬─────────┘
                  ▼
       GPU-specific executable
          such as a cubin
                  │
                  ▼
        native GPU instructions
             called SASS
                  │
                  ▼
             NVIDIA GPU
```

Not every tool exposes every intermediate stage to the user. Libraries may
ship precompiled binaries, and higher-level systems may manage compilation and
kernel selection internally.

## 8. Offline compilation versus driver JIT

There are two common ways to obtain GPU-specific executable code:

### Offline compilation

`ptxas` compiles PTX into code for a specified `sm_*` target before the
application runs.

Advantages include:

- no PTX compilation cost during first execution;
- predictable target architecture;
- compiler diagnostics available during the build.

### Driver JIT compilation

An application can ship PTX, and the installed NVIDIA driver can compile it for
the available compatible GPU.

Advantages include:

- some portability across later compatible GPU architectures;
- the installed driver can target the actual GPU.

The first load may incur JIT overhead, although generated code can be cached.

## 9. When should you write Triton?

Triton is a strong candidate when:

- the operation is naturally tensor-shaped;
- several simple operations can be fused;
- memory traffic is the likely bottleneck;
- existing PyTorch or CUDA libraries do not provide the required fused
  operation;
- development speed matters;
- you want a custom operation that fits naturally beside TorchInductor.

CUDA C++ remains the stronger choice when:

- you need precise synchronization or memory control;
- the algorithm has irregular parallelism;
- you need a hardware feature Triton does not expose;
- the kernel must integrate with an existing CUDA C++ codebase;
- you are building more than a tensor kernel.

## Final takeaway

```text
CUDA
    The complete NVIDIA GPU-computing platform.

CUDA C++
    A low-level and highly controllable way to write GPU programs.

Triton
    A higher-level way to write many tensor-oriented GPU kernels.

PTX
    NVIDIA's virtual GPU instruction set.

cubin and SASS
    GPU-specific executable packaging and native instructions.
```

Triton can replace handwritten CUDA C++ for some kernels, but it does not
replace CUDA as a platform. Choose between Triton and CUDA C++ based on the
algorithm, required control, development cost, and measured performanceâ€”not on
the assumption that one is universally faster.
