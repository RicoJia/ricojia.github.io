---
layout: post
title: "[Robotics] Two Paths from PyTorch to Fast GPU Inference"
date: 2026-07-19 13:19
subtitle: TensorRT, Torch Compile
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - robotics
---
# `torch.compile` and TensorRT

A PyTorch model can reach an NVIDIA GPU through several optimization paths.
Two important choices are:

- compile PyTorch execution with **`torch.compile` and TorchInductor**;
- convert the inference graph into a **TensorRT engine**.

Both optimize computation graphs, but they have different goals, constraints,
and deployment models.

```
PyTorch model
     ├──► torch.compile + TorchInductor
     └──► TensorRT
```

This article explains what happens along each path and how to choose between
them.

## 1. The short mental model

- torch.compile: PyTorch's high-level compilation entry point.
- TorchDynamo: Captures supported PyTorch operations from Python execution.
- AOTAutograd: Captures and transforms forward and backward computations.
- TorchInductor: PyTorch's default optimizing compiler backend.
- Triton language/compiler: Generates many of the GPU kernels used by TorchInductor.
- TensorRT: Builds optimized inference engines for NVIDIA GPUs.
- NVIDIA Triton Inference Server: Serves TensorRT engines and other model formats over a network.

The last two uses of *Triton* are unrelated products. The Triton language is a
GPU-kernel compiler; NVIDIA Triton Inference Server is a model-serving system.

## 2. How `torch.compile` works

`torch.compile` is the user-facing PyTorch entry point:

```python
compiled_model = torch.compile(model)
```

Its default compilation path can be simplified as:

```
Python / PyTorch program
          │
          ▼
     TorchDynamo
captures graph regions and creates guards
          │
          ├── inference ───────────────────────────────┐
          │                                           │
          └── training ──► AOTAutograd ───────────────┤
                                                      ▼
                                               TorchInductor
                                         optimization and lowering
                                                      │
                             ┌────────────────────────┴─────────────────┐
                             ▼                                          ▼
                  Generated GPU kernels                      Optimized libraries
                  often using Triton                       and custom operators
                             │                                          │
                             └────────────────────┬─────────────────────┘
                                                  ▼
                                          GPU execution
```

PyTorch documents TorchInductor as the default `torch.compile` backend and
Triton as a key GPU code-generation building block: [PyTorch compiler documentation](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html).

Not every operation becomes a Triton kernel. Inductor may:

- generate another type of kernel;
- call cuBLAS, cuDNN, or another optimized library;
- preserve a registered custom operator;
- place unsupported code outside the compiled region.

### 2 - 1 Examples : kernel fusion

Consider:

```python
def operation(x):
    y = x + 1
    z = torch.relu(y)
    return z * 2
```

Eager execution may conceptually launch three kernels:

```text
kernel 1: add 1       â†’ write y
kernel 2: ReLU        â†’ write z
kernel 3: multiply 2  â†’ write result
```

`TorchInductor` may combine them into one generated kernel:

```
load x → add 1 → ReLU → multiply by 2 → store result
```

Fusion can reduce:

- kernel-launch overhead;
- framework dispatch overhead;
- intermediate memory reads and writes.

This is one reason a compiler can improve performance without changing the
model's mathematical definition.

Similarly, fuse ordinary PyTorch operations

```python
def preprocess(x):
    x = (x - mean) / std
    return torch.clamp(x, 0, 1)
```

Eager PyTorch may launch several kernels:

```text
subtract → divide → clamp
```

TorchInductor may generate one fused Triton kernel:

```text
load x → subtract mean → divide by std → clamp → store
```

This is useful for custom PyTorch logic, including preprocessing and postprocessing—not just standard neural-network layers.

### 2 - 2 Graph breaks

`torch.compile` does not necessarily capture an entire Python program as one
graph.

For example:

```python
def forward(x):
    if x.sum().item() > 0:
        print("positive")
    return model(x)
```

The tensor-to-Python conversion, data-dependent Python branch, and `print()` can
cause a **graph break**:

```
compiled region 1
      ↓
Python execution
      ↓
compiled region 2
```

Graph breaks do not necessarily change correctness, but they reduce the
compiler's optimization scope and may add overhead.

### 2 - 3 Guards and recompilation

`TorchDynamo` creates **guards** for assumptions made while capturing a graph.
Examples include:

- input device is CUDA
- input dtype is `float16`
- input shape satisfies captured constraints
- relevant model and Python state has not changed

**If a guard fails, PyTorch may compile another graph variant**. Frequently changing
shapes or Python state can therefore cause repeated compilation.

This matters in vision pipelines where:

- **crop sizes vary;**
- **batch sizes change;**
- different model branches execute;
- input dtypes or devices change.

Stable shapes generally make compilation and benchmarking easier.

### 2 - 4 Why the first call is slow

The first compiled call may perform:

- graph capture;
- graph lowering;
- kernel generation;
- **kernel compilation**;
- **optional autotuning**;
- cache creation.

Later executions can **reuse compiled results**.

Benchmark cold-start and steady-state latency separately:

```python
compiled_model = torch.compile(model)

with torch.inference_mode():
    for _ in range(10):
        compiled_model(example_input)

torch.cuda.synchronize()

# Measure warmed execution here.
```

Because CUDA execution is asynchronous, **synchronize immediately before and**
**after the timed region when measuring GPU latency**.

## 3. What TensorRT does

TensorRT is an **inference optimizer and runtime** for NVIDIA GPUs. It consumes a
supported inference graph and builds a serialized engine.

TensorRT optimization can include:

- **graph and layer fusion**;
- **precision selection**;
- tactic selection;
- **tensor-layout** decisions;
- **memory planning**;
- optimization for specified input-shape profiles.

A simplified workflow is:

```
PyTorch inference model
           │
           ├──► Torch-TensorRT
           │
           └──► export path such as ONNX
                         │
                         ▼
                  TensorRT network
                         │
                         ▼
                  TensorRT builder
             optimization and tactic selection
                         │
                         ▼
               serialized TensorRT engine
                         │
                         ▼
                  TensorRT runtime
                         │
                         ▼
                    NVIDIA GPU
```

Unlike `torch.compile`, TensorRT is inference-focused. Its engine is a deployment
artifact constrained by factors such as:

- supported operators;
- allowed input shapes;
- selected precisions;
- target hardware and software compatibility;
- custom plugin availability.

See the current [NVIDIA TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html).

### 3-1 TensorRT example: optimize a convolution block

Given:

```text
Convolution → Bias → BatchNorm → ReLU
```

TensorRT can:

1. Fold BatchNorm parameters into the convolution weights and bias.
2. Fuse the remaining operations where supported.
3. Choose the fastest convolution implementation, called a **tactic**, for the GPU and input shape.
4. Run it in FP16 or INT8 when configured and supported.

The resulting engine may effectively execute:

```text
one optimized FP16 convolution block
```

### 3-3 Torch-TensorRT

`Torch-TensorRT` connects PyTorch with TensorRT. Depending on the workflow, it can
compile supported model regions for TensorRT and integrate them with PyTorch
execution.

PyTorch also documents an inference backend registered by Torch-TensorRT:

```python
import torch_tensorrt

compiled_model = torch.compile(
    model,
    backend="tensorrt",
)
```

This is different from the default:

```python
compiled_model = torch.compile(
    model,
    backend="inductor",
)
```

`torch.compile` is the front-end entry point in both examples, but the selected
backend determines how captured graphs are lowered and executed.

## 3-4 TorchInductor versus TensorRT

| Question | TorchInductor | TensorRT |
| --- | --- | --- |
| Primary input | Captured PyTorch graph | Supported inference graph |
| Typical entry point | `torch.compile(..., backend="inductor")` | TensorRT APIs, export tools, or Torch-TensorRT |
| Training support | Yes | No; inference only |
| Typical result | Compiled graph regions and generated kernels | Serialized inference engine |
| Python integration | Remains relatively close to PyTorch | Creates a more constrained deployment boundary |
| GPU execution | Generated kernels and library calls | Selected TensorRT tactics, plugins, and library implementations |
| Main strength | Low-friction PyTorch optimization | Aggressive NVIDIA inference optimization and deployment |

They occupy similar **graph-optimization positions**, but their inputs, outputs,
supported workloads, and runtime models are different:

```
PyTorch graph
    ├──► TorchInductor ──► generated kernels and library calls
    └──► TensorRT ───────► optimized inference engine
```

## 4. Which one should you use?

```text
torch.compile
    PyTorch's high-level compilation entry point - optimizes executable PyTorch code and can generate kernels for custom combinations of tensor operations.

TorchInductor
    The default compiler backend, often generating Triton GPU kernels.

TensorRT
    An NVIDIA inference optimizer and runtime that builds deployable engines. Optimizes a supported inference graph and selects highly tuned NVIDIA implementations for its layers.

NVIDIA Triton Inference Server
    A serving system that can host TensorRT engines and other model formats.
```

For a stable production inference graph, `TensorRT` is often the strongest candidate. `torch.compile` is attractive when you want optimization while retaining PyTorch flexibility. The correct decision comes from measuring the complete application, not merely comparing isolated model calls. A practical example could be:

```text
torch.compile:
    Fuse crop normalization, masking, and tensor transformations.

TensorRT:
    Optimize and package the FoundationPose scorer/refiner network
    into a fixed inference engine.
```
