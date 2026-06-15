---
layout: post
title: "[CUDA - 6] CUDA Inference Speedups"
date: 2026-01-21 13:19
subtitle: inference mode, jit tracing, cuda graphs, tensor rt, optimize_for_inference
comments: true
header-img: img/post-bg-alibaba.jpg
tags:
  - CUDA
---
## Introduction

I built a RF-DETR + FoundationPose pipeline. the clean optimization stack is:

1. Make inference measurable. Log per-stage latency: detector, register, track, publish, visualize.
2. Use inference mode and FP16 carefully. Validate that pose quality does not degrade too much.
3. Use RF-DETR optimize_for_inference for fixed-shape detection. Treat it as JIT tracing/export-style inference, not TensorRT.
4. Warm up the real inference path. First-frame latency is not steady-state latency.
5. Consider CUDA Graphs for repeated fixed-shape tracking calls. Especially refiner/scorer forward passes.
6. Use TensorRT for stable neural-network subgraphs. Refiner and scorer are better candidates than the whole ROS node.
7. Keep `empty_cache()` out of the hot path. Reusing cached GPU memory usually helps repeated inference.
8. Do not ignore ROS overhead. Debug image publishing, queue backpressure, camera-info synchronization, and visualization can dominate the end-to-end latency.

The main lesson is that “model optimization” and “system optimization” are not the same thing. JIT tracing, CUDA Graphs, and TensorRT can make the neural network faster, but the full pose-estimation system only becomes responsive when the ROS pipeline also stops blocking on old frames, debug images, and visualization work.

## optimize_for_inference()

`optimize_for_inference()` is basically “Make an inference-only copy of the model, put it in eval/export mode, optionally cast to FP16, then trace the forward path for a fixed input shape.”

Before optimization, my RF-DETR timing was roughly (avg / max):

```text
predict = 145 / 265 ms
```

After:

```python
model.optimize_for_inference(
    compile=True,
    batch_size=1,
    dtype=torch.float16,
)
```

the timing became roughly:

```text
predict = 67 / 86 ms
```

So the win likely came from a combination of:

- eval/export forward path
- FP16 inference
- fixed-shape assumptions
- JIT traced graph
- less Python overhead
- more stable GPU execution path

### JIT tracing

In an un-optimized piece of code, you might see something like:

```python
def forward(x):
    x = patch_embed(x)
    x = backbone(x)
    x = transformer_encoder(x)
    x = transformer_decoder(x)
    boxes, logits = detection_head(x)
    return boxes, logits
    
```

That Python code calls many PyTorch ops. Each call goes through Python, PyTorch dispatch, CUDA kernel selection, and GPU execution.

```python
Python interpreter
    ↓
nn.Module.__call__
    ↓
hooks / mode checks / wrapper logic
    ↓
Python attribute lookup
    ↓
PyTorch op dispatch
    ↓
CUDA/cuDNN/cuBLAS kernel launch
    ↓
GPU does the actual work
```

After JIT tracing, the model is more like:

```text
aten::conv2d
aten::relu
aten::reshape
aten::matmul
aten::softmax
aten::linear
...
```

So instead of interpreting the original Python `forward()` path every time, **PyTorch can run the recorded tensor graph**. That is why tracing can improve inference speed: the model path becomes more static and less Python-heavy. JIT tracing means: **run the model once with an example input**, watch **which tensor operations happen**, and **save that sequence of operations as a TorchScript graph in PyTorch's C++ runtime**.

`model.optimize_for_inference(compile=True, ...)` does NOT do TensorRT or PyTorch `torch.compile`. `compile=True` is closer to:

```python
inference_model = deepcopy(model)
inference_model.eval()
inference_model.export()
inference_model = inference_model.to(dtype=torch.float16)

dummy = torch.randn(
    batch_size,
    3,
    resolution,
    resolution,
    dtype=torch.float16,
    device="cuda",
)

traced_model = torch.jit.trace(inference_model, dummy)
```

But tracing has a major limitation: **it records what happened for the example input**. It does **not understand arbitrary Python logic**. It follows the path taken during the trace. For example:

```python
class ToyModel(torch.nn.Module):
    def forward(self, x):
        if x.mean() > 0:
            return x * 2
        else:
            return x - 2
```

If I trace this with a positive dummy input:

```python
dummy = torch.ones(1, 3)
traced = torch.jit.trace(model, dummy)
```

The traced graph may effectively remember the `x * 2` path. Later, if I pass an input where `x.mean() <= 0`, the traced model may still follow the originally recorded path instead of switching branches. That is why JIT tracing is best for models that are mostly tensor operations **without condition branches**. In cases like RF-DETR inference, this is usually acceptable because the model’s forward pass is mostly a fixed neural network graph:

```text
image tensor
    ↓
backbone
    ↓
transformer
    ↓
box/class heads
    ↓
boxes + logits
```

But tracing is specific to a set of fixed assumptions used during tracing:

```text
same batch size
same input resolution
same dtype
same model mode
same forward path
```

---

## CUDA Graphs

CUDA Graphs solve a different problem. Even if the model is already fast, a GPU inference call may launch many small CUDA kernels:

```text
kernel 1: convolution
kernel 2: activation
kernel 3: normalization
kernel 4: reshape
kernel 5: matrix multiply
kernel 6: softmax
kernel 7: matrix multiply
...
```

Each kernel launch has overhead. The CPU has to prepare the launch, pass arguments, interact with the CUDA driver, and submit work to the GPU. If the model has many small operations, this launch overhead can become significant.

A CUDA Graph is basically “this sequence of CUDA work happens every frame. Capture it once, then replay the same GPU work each frame.” Conceptually:

```text
Normal eager execution:
    Python launches kernel A
    Python launches kernel B
    Python launches kernel C
    Python launches kernel D

CUDA Graph replay:
    Python says: replay graph
    CUDA launches the captured sequence
```

The big constraint is that the replay must use the same structure each time:

- same operations
- same tensor shapes
- same memory addresses
- same control flow

CUDA Graph replay uses the same memory addresses captured earlier. This means the input and output tensors need to be long-lived. Instead of allocating a new input tensor every frame, we copy new frame data into the same preallocated tensor. A simplified inference-only CUDA Graph flow looks like this:

```python
model.eval()

static_input = torch.empty(
    1, 3, 160, 160,
    device="cuda",
    dtype=torch.float16,
)

# Warm up first.
# This lets CUDA, cuDNN, cuBLAS, and PyTorch allocator paths initialize.
for _ in range(3):
    with torch.inference_mode():
        static_output = model(static_input)

torch.cuda.synchronize()

# Capture.
graph = torch.cuda.CUDAGraph()

with torch.cuda.graph(graph):
    static_output = model(static_input)

# Replay every frame.
def infer(new_input):
    static_input.copy_(new_input)
    graph.replay()
    return static_output
```

The key trick is here:

```python
static_input.copy_(new_input)
graph.replay()
```

We are not giving the graph a new tensor. We are copying new data into the old tensor’s memory, then replaying the captured graph. **The copy has a cost, yes. But CUDA Graphs try to save more than that cost by avoiding repeated CPU-side launch overhead:**

```
for every frame:  
 launch conv kernel  
 launch relu kernel  
 launch matmul kernel  
 launch softmax kernel  
 launch matmul kernel  
 launch linear kernel
```

This is why CUDA Graphs are attractive for FoundationPose tracking. In tracking mode, the same refiner/scorer network may run every frame with mostly fixed tensor shapes. That is exactly the kind of workload CUDA Graphs like.

For example:

```text
FoundationPose tracking path:
    previous pose
    RGB crop
    depth crop
    rendered object hypothesis
    refiner network
    scorer network
    updated pose
```

If the crop size, batch size, and number of pose hypotheses stay fixed, the neural-network part can become graph-friendly.

But not every part of FoundationPose is graph-friendly. Some parts may have **dynamic shape, CPU-side logic, rendering calls, synchronization, or conditional control flow**. A safer optimization is to graph only the stable neural network submodule:

```text
Do not graph everything:
    camera callback
    ROS message conversion
    detection lookup
    mask creation
    debug image publishing
    conditional registration logic

Maybe graph:
    fixed-shape refiner forward pass
    fixed-shape scorer forward pass
```

---

## TensorRT

TensorRT is another level deeper.

First, inference means running a trained model forward to get predictions. No gradient calculation, no weight updates, no backpropagation. **TensorRT is an inference compiler** and runtime for NVIDIA GPUs. Instead of running the model through normal PyTorch, we export a **stable subgraph, usually through ONNX**, and let **TensorRT build a GPU-specific** engine.

```text
PyTorch model
    ↓
ONNX export
    ↓
TensorRT builder
    ↓
optimized TensorRT engine
    ↓
TensorRT runtime inference
```

For FoundationPose, the obvious candidates are the refiner and scorer networks:

```text
refiner_model.onnx
score_model.onnx
```

TensorRT reads the model graph and optimizes it for the target GPU. It may:

- fuse layers
- choose faster kernels
- choose memory layouts
- pre-plan GPU memory
- use FP16 or other lower precision modes
- remove unnecessary runtime flexibility

For example, a PyTorch graph might contain:

```text
Conv
BatchNorm
ReLU
```

TensorRT may turn that into something closer to:

```text
fused Conv + BatchNorm + ReLU kernel
```

The math is intended to be equivalent, but the execution is different. Fewer kernel launches and fewer intermediate memory writes can reduce latency.

Another example:

```text
matrix multiply
activation
reshape
linear layer
```

TensorRT may choose a different GEMM tactic, tensor layout, or Tensor Core path depending on:

```text
GPU model
CUDA version
TensorRT version
enabled precision
workspace size
input shape profile
```

That is why TensorRT engines are not just “the same model file.” They are built for a deployment configuration.

A simplified TensorRT build flow looks like this:

```python
# 1. Export from PyTorch to ONNX.
torch.onnx.export(
    model,
    example_inputs,
    "refiner.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=None,
)

# 2. Build TensorRT engine from ONNX.
builder = trt.Builder(logger)
network = builder.create_network(...)
parser = trt.OnnxParser(network, logger)

with open("refiner.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)

engine = builder.build_serialized_network(network, config)

# 3. Run inference with TensorRT runtime.
context = engine.create_execution_context()
context.execute_async_v3(...)
```

This is not exact production code, but it shows the idea:

```text
export graph
build engine
run engine
```

TensorRT is useful when the model subgraph is stable enough to justify a separate compiled engine. It is less attractive for messy code that mixes Python control flow, ROS logic, rendering, dynamic shapes, and CPU/GPU synchronization.

So for FoundationPose, the right strategy is not:

```text
TensorRT everything.
```

It is more like:

```text
Find the repeated neural-network submodules.
Export those.
Build TensorRT engines for those fixed shapes.
Keep the rest in Python/PyTorch/ROS.
```

That is why TensorRT for the refiner and scorer makes sense, but TensorRT for the entire pose-estimation node may not.

---

## `torch.cuda.empty_cache()`

I also tested removing a `torch.cuda.empty_cache()` call. The important correction is:

```text
empty_cache() does not free live tensors.
```

If a tensor is still being used, `empty_cache()` cannot delete it. What it can do is release unused cached memory blocks held by PyTorch’s caching allocator back to CUDA.

In normal repeated inference, keeping the cache is often good. The model tends to allocate similar tensor shapes every frame. If PyTorch keeps those memory blocks around, it can reuse them quickly.

The bad pattern is:

```python
while True:
    output = model(input)
    torch.cuda.empty_cache()
```

That can make latency worse because every frame may force PyTorch/CUDA to reacquire memory that could have been reused.

The better pattern is:

```python
# During steady-state inference:
with torch.inference_mode():
    output = model(input)

# Do not call empty_cache() in the hot path.
```

I would only consider `empty_cache()` for cases like:

```text
after a one-time huge allocation
between different phases of a program
when another process needs GPU memory
when debugging memory fragmentation
```

But for a real-time pose pipeline, calling it every frame is usually not an optimization. It is more likely to add allocator overhead and increase jitter. `empty_cache()` can make `nvidia-smi` look cleaner

---

### cudnn benchmark

`torch.backends.cudnn.benchmark = True` tells PyTorch/cuDNN to try multiple GPU convolution algorithms and cache the fastest one for the input shapes it sees. This can speed up CNN-style models when input sizes are mostly fixed, but can hurt performance if shapes change constantly because cuDNN may keep re-benchmarking.

### inference mode()

`torch.inference_mode()` disables autograd bookkeeping during inference. Compared with normal execution, PyTorch does not build gradient graphs, and it also disables extra tracking such as view tracking and version counter updates. That usually means **lower memory use** and **faster execution** for pure prediction code. PyTorch notes that `inference_mode` does **not** automatically call `model.eval()`, so you still need `model.eval()` for dropout/batchnorm behavior.
