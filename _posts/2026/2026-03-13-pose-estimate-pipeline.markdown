---
layout: post
title: Computer Vision - Pose Estimate Pipeline
date: '2026-03-13 13:19'
subtitle: FP16, BF16, Mixed Precision Training
comments: true
header-img: img/post-bg-infinity.jpg
tags:
    - Deep Learning
---

## **Building a 2D-to-6D Object Pose Pipeline with RF-DETR and FoundationPose**

I built a small pose-estimation pipeline that combines RF-DETR and FoundationPose. RF-DETR handles the 2D detection problem: find the target object in the camera image. FoundationPose handles the 6D pose problem: given RGB-D data, camera intrinsics, and the object CAD model, estimate the object pose relative to the camera. The first version worked, but it was too slow and too bursty. The rest of the project became a latency investigation: which parts were neural network inference, which parts were GPU warmup, and which parts were just ROS pipeline backpressure?


```
ZED / RealSense
    ↓
RGB image + depth image + camera_info
    ↓
RF-DETR
    ↓
2D bbox + class + confidence
    ↓
class filter: phone / target object
    ↓
mask or crop for FoundationPose
    ↓
FoundationPose + CAD model + camera intrinsics
    ↓
6D pose + 3D bounding box
    ↓
ROS pose topic + visualization node
```

RF-DETR does **2D object detection**, not 3D pose. Its job is to say: “the target object is probably here.” FoundationPose then uses RGB-D, camera intrinsics, and the CAD/reference object representation to estimate or track 6D pose. FoundationPose’s paper describes it as supporting pose estimation and tracking for novel objects, given a CAD model or reference images.

A bounding box alone is usually not the cleanest input for FoundationPose initialization. For registration, you generally want a **segmentation mask** or a reliable object region. A bbox can be used to generate/crop a mask, but the post should say that clearly. Define the coordinate frames early:

```
object frame: CAD/model coordinate frame
camera frame: RGB-D camera coordinate frame
world/base frame: optional robot or map frame
```

Then state the output precisely:

```
T_camera_object
```

Your baseline FoundationPose logs showed roughly:

```
estimate_pose ≈ 405 / 335 ms
register_pose ≈ 3962 / 4445 ms
track_pose    ≈ 300 / 325 ms
```

Then FP16/inference-mode work brought tracking closer to: `track_pose ≈ 200 / 229 ms` 
Later TensorRT and seed changes got tracking down further, TODO: with the “scorer + two seeds” case around:  
```
track_pose ≈ 136 / 163 ms
```

### **What RF-DETR `optimize_for_inference()` Actually Does**

In RF-DETR, `compile=True` is a library-specific flag. In the current implementation, it prepares an inference-only model and uses `torch.jit.trace` when compilation is requested. That makes inference faster for a fixed batch size, image resolution, and dtype, but it is not TensorRT and not PyTorch `torch.compile`.

```python
model.optimize_for_inference(
    compile=True,
    batch_size=1,
    dtype=torch.float16,
)
```

The above creates this effect:

```
Before:
predict = 145 / 265 ms

After optimize_for_inference:
predict = 67 / 86 ms
frame   = 37 / 78 ms
```

The RF-DETR inference optimization docs/source summary describe `compile=True` as using a dummy input and `torch.jit.trace`, then storing optimized batch size and resolution metadata for prediction-time validation. PyTorch’s tracing flow records operations from example inputs, and tracing does not capture arbitrary Python control flow the way scripting does. So basic distinctions are:
- Normal PyTorch model: Python forward path + flexible shape + runtime dispatch
- RF-DETR optimized model: eval mode + export-style forward + fixed dtype + traced graph

`model.eval()` disables dropout and makes batch norm use stored statistics, but it does **not** guarantee fully deterministic GPU execution. CUDA/cuDNN kernels, FP16 math, and graph-level rewrites can still produce tiny numerical differences.


### **Why the First Few Frames Are Slow: Warming Up RF-DETR and FoundationPose**


Loaded cached JIT model ≠ GPU fully warmed up. Even after loading a cached/traced model, first real inference can still pay for CUDA context setup, allocator behavior, cuDNN algorithm selection, kernel loading, graph executor warmup, and postprocess synchronization. Your notes explicitly observed a large gap between the first frames and steady state, then added dummy warmup inference to reduce confusion. 

```python
def _warmup_model(self):
    resolution = int(getattr(self.model.model, "resolution", 560))
    image = np.zeros((resolution, resolution, 3), dtype=np.uint8)

    for _ in range(2):
        self._predict_from_cv_image(image)
```
Warmup should match the real inference path as closely as possible: same resolution, same dtype
same batch size, same device, same postprocess path if possible

**FoundationPose Performance: Registration Is Expensive, Tracking Is the Fast Path. Generally, the faster tracking, the less likely to lose target**

register:
- use RGB-D + mask + CAD/model information
- search/refine initial pose
- expensive

track:
- start from previous pose
- refine pose for the current frame


## JIT tracing

TODO: what it does exactly, with pseudo code

## CUDA Graphs

TODO: Whatis a CUDA graph, 

capture repeated CUDA work and replay it with less launch overhead; same operations, same shapes, same memory pattern every frame. PyTorch’s CUDA Graph guidance says to warm up the workload before capture and keep long-lived input/output tensors because replay uses the same memory addresses.


## TensorRT

What is inference 

build a GPU-specific inference engine from an ONNX/PyTorch-style graph.  
the model subgraph is stable and worth compiling separately

NVIDIA describes TensorRT as an SDK for optimizing and accelerating inference on NVIDIA GPUs, with support for PyTorch/ONNX-style import paths, mixed precision, dynamic shapes, and GPU-specific optimizations.
Your `torch.cuda.empty_cache()` note is right. It does not free live tensors. PyTorch documents it as releasing unoccupied cached allocator memory so other GPU applications can use it, and notes that it does not increase the amount of GPU memory available to PyTorch.


## **The Biggest Speedup Was Not a Model Trick: It Was the Pipeline**

publishing debug images cost around 45–60 ms in some runs, and the final improvement came after decoupling camera info and fixing visualization queue behavior. The model can be fast, but the ROS graph can still feel slow if:

- camera_info is synchronized too strictly
- debug images are published in the hot path
- visualization queues back up
- inference and rendering share the same callback path
- old frames are processed instead of dropped

TensorRT helped the model path, but the end-to-end pipeline became usable only after I removed visualization/backpressure from the critical path and fixed frame synchronization behavior.

inference node:
    subscribe RGB-D + camera_info
    run RF-DETR / FoundationPose
    publish compact pose messages

visualization node:
    subscribe pose + optional image
    draw overlays
    allowed to drop frames
    
The inference node should publish small messages:

```
PoseStamped
Detection2D/3D
TransformStamped
small debug metadata
```
