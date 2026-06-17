---
layout: post
title: Computer Vision - Pose Estimate Pipeline
date: '2026-03-13 13:19'
subtitle: 
comments: true
header-img: img/post-bg-infinity.jpg
tags:
    - Deep Learning
---

## **Building a 2D-to-6D Object Pose Pipeline with RF-DETR and FoundationPose**

I built a small pose-estimation pipeline that combines RF-DETR and FoundationPose. RF-DETR solves the 2D detection problem: find the target object in the camera image. FoundationPose solves the 6D pose problem: given RGB-D data, camera intrinsics, and an object CAD model, estimate the object pose relative to the camera.

The first version worked, but it was too slow and too bursty. What started as a model integration project turned into a latency investigation: which parts were neural network inference, which parts were GPU warmup, and which parts were ROS pipeline backpressure?

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

The coordinate frames are:

object frame: CAD/model coordinate frame
camera frame: RGB-D camera coordinate frame
world/base frame: optional robot or map frame

The main output of the pose estimator is:

T_camera_object

That transform represents the pose of the object frame relative to the camera frame.

RF-DETR does not estimate 3D pose. Its job is only to localize the target object in the image. FoundationPose then uses RGB-D, camera intrinsics, and the CAD model or reference object representation to estimate or track the 6D pose.

A bounding box alone is not usually the cleanest input for FoundationPose initialization. For registration, a segmentation mask or reliable object region is better. In this pipeline, the detector provides the object location, and that detection is used to build or crop the region passed into the pose-estimation stage.

## Baseline Performance

The baseline FoundationPose logs showed that registration was much more expensive than tracking:

| Stage         |   Baseline latency |
| ------------- | -----------------: |
| estimate_pose |   about 335–405 ms |
| register_pose | about 3962–4445 ms |
| track_pose    |   about 300–325 ms |

This matters because registration and tracking are different modes. Registration tries to find the initial pose from RGB-D, a mask, and the object model. Tracking starts from the previous pose and refines it for the current frame.

Registration is the expensive recovery path. Tracking is the fast path.

After FP16 and inference-mode changes, tracking improved to roughly:

```text
track_pose ≈ 200 / 229 ms
```

Later TensorRT and seed changes brought tracking down further, with the scorer plus two-seed case around:

```text
track_pose ≈ 136 / 163 ms
```

Lower tracking latency reduces the gap between frames, which can make the system more stable when the camera or object is moving.

## What RF-DETR `optimize_for_inference()` Actually Does

In RF-DETR, `compile=True` is a library-specific inference optimization flag. In the implementation I used, it prepares an inference-only model and uses `torch.jit.trace` when compilation is requested.

```python
model.optimize_for_inference(
    compile=True,
    batch_size=1,
    dtype=torch.float16,
)
```

This made a large difference in the detector path:

| RF-DETR mode                     |                Latency |
| -------------------------------- | ---------------------: |
| Before optimization              | predict ≈ 145 / 265 ms |
| After `optimize_for_inference()` |   predict ≈ 67 / 86 ms |
| Frame path after optimization    |     frame ≈ 37 / 78 ms |

The important distinction is that this is not TensorRT and it is not PyTorch `torch.compile`. In this case, the optimization path is closer to an export-style inference path with fixed assumptions about batch size, resolution, and dtype.

A useful way to think about it:

```text
Normal PyTorch model:
    Python forward path
    flexible shapes
    runtime dispatch

RF-DETR optimized model:
    eval mode
    fixed dtype
    fixed batch/resolution assumptions
    traced graph
```

`model.eval()` is still important because **it disables dropout** and makes batch normalization use stored statistics. But `eval()` does not guarantee perfectly deterministic GPU execution. CUDA kernels, cuDNN behavior, FP16 math, and graph-level rewrites can still produce small numerical differences.

## Why the First Few Frames Are Slow

**Loading a cached traced model does not mean the GPU is fully warmed up.**

The first real inference can still pay for CUDA context setup, allocator behavior, cuDNN algorithm selection, kernel loading, graph executor warmup, and postprocess synchronization. In my logs, the first few frames were noticeably slower than steady state, so I added dummy warmup inference to avoid confusing startup effects with actual runtime latency.

```python
def _warmup_model(self):
    resolution = int(getattr(self.model.model, "resolution", 560))
    image = np.zeros((resolution, resolution, 3), dtype=np.uint8)

    for _ in range(2):
        self._predict_from_cv_image(image)
```

Warmup should match the real inference path as closely as possible:

```text
same resolution
same dtype
same batch size
same device
same preprocessing
same postprocessing, if possible
```

Otherwise the warmup can make the logs look cleaner without actually warming the path that matters.

## The Biggest Speedup Was Not a Model Trick

The most important lesson was that the model path was only part of the system.

Publishing debug images cost around 45–60 ms in some runs. The final usability improvement came from decoupling camera info handling and fixing visualization queue behavior.

The model can be fast, but the ROS graph can still feel slow if:

* `camera_info` is synchronized too strictly
* debug images are published in the hot path
* visualization queues back up
* inference and rendering share the same callback path
* old frames are processed instead of dropped

The better architecture was to keep inference and visualization separate.

```text
inference node:
    subscribe RGB-D + camera_info
    run RF-DETR / FoundationPose
    publish compact pose messages

visualization node:
    subscribe pose + optional image
    draw overlays
    allowed to drop frames
```

The inference node should publish small messages:

```text
PoseStamped
Detection2D / Detection3D
TransformStamped
small debug metadata
```

The visualization node can publish images, overlays, and debugging views, but it should not be allowed to block the pose-estimation path.

TensorRT helped the model path, but the end-to-end pipeline became usable only after visualization and ROS backpressure were removed from the critical path.
