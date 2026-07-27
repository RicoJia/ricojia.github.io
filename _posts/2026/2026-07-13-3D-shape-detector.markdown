---
layout: post
title: "[Robotics] 3D-shape-detection-from-point-clouds"
date: 2026-07-13 13:19
subtitle: CenterPoint
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - robotics
---
## BEV

BEV is full 7 dof representation: `(x,y,z,l, w, h, yaw)`.  Its scene is a heatmap on a top down grid, and where it localizes its peak. Here, (x,y,z) is a BEV pillar, which is the 2.5D representation of an object

The echo scope gives a 128x128 grid already, which is a quasi 2.5D environment - we can represent the world on a (x,y) grid with height instead of the full (x,y,z) representation. In some cases we might have multiple objects at different z heights on the same (x,y), that's why the sea environement is not strictly 2.5D, but that's not very common.

| Representation                | Detector                  | Keeps full z res?        | Cost / Risk                                    |
| ----------------------------- | ------------------------- | ------------------------ | ---------------------------------------------- |
| Dense BEV pillars             | CenterPoint heatmap       | z regressed, not gridded | Cheap, stable, mature                          |
| Dense/sparse 3D voxels        | 3D center head            | Yes                      | spconv/Minkowski build fragility, more compute |
| Raw points                    | PointNet++ + vote/cluster | Yes                      | Slow FPS, poor fit for gridded input           |
| Range-view (native beam grid) | 2D CNN / DETR             | z implicit via range     | Scale varies with range                        |

In comparison, we are not using PointNet++ because:

- its FPS and ball-query are built for sparse, large-extent lidar points. FPS is O(N^2), whereas Echoscope's grid already has a  uniform density.
- Its a backbone, not a detector. Internally, PointNet++ has a voting (votenet), clustering, per-point proposals (PointRCNN). More complex than Centerpoint

The network architecture is:

```text
BEV input
   ↓
small U-Net-style convolutional backbone
   ↓
CenterPoint-style center heatmap and box-regression heads
```

### Example

The CenterPoint produces two outputs:

- Center heatmap: [1, 64, 64]
- Box regression: [8, 64, 64]

The CenterPoint-style ideas are:

1. **Object centers are heatmap peaks.**
2. **The box is regressed at the center location.**
3. **Local peak NMS is applied to the heatmap.**
4. **Regression loss is evaluated only at ground-truth centers.**

For example: the network predicts an object center:

$$  
[  
\Delta x,\Delta y,z,  
\log l,\log w,\log h,  
\sin(2\theta),\cos(2\theta)  
].  
$$
The code predicts the heat map directly:

```python
return self.hm(f), self.reg(f)
```

and during decoding:

```python
keep = (max_pool2d(p, 3, 1, 1) == p)
```

which keeps local heatmap peaks. During training, regression values are selected only at the ground-truth center:

```python
pr_c = pr[
    torch.arange(B),
    :,
    pos[:, 0],
    pos[:, 1],
]
```

---

## Model Training Experience

We actually tried creating hand-crafted features, then feeding them into the BEV network. It was basically early-stage work that helped us identify key elements in noise for data synthesis and quickly bring up a semi-working model.

1. Debuggability.
 1. Hand-crafted 8-channel features (p90 height, relief, shadow density, etc.) are interpretable
  1. every diagnostic this session ran (debris-field FP root-cause, "removing height collapses the detector," real-negative crushing weak signal) worked because I could point at a specific channel and reason about it. **This indeed helped with ablation-testing with noise synthesis**
  2. The root causes it actually diagnosed are properties of the data, not obviously fixable by a better encoder.  
   1. the ~9-11x real-domain noise gap,
   2. real-negative training crushing the weak close-range signal
   3. the need for real background noise
 2. A learned encoder is a black box until it's trained; early on you can't tell if a failure is architecture, data, or training. But it turned out to be better :)
 3. The PointPillars README said this explicitly going in:  It could easily have just reproduced the same failure with extra steps.
2. Faster iteration loop.
 1. Hand-crafted features get precomputed once into features.bin, so training is a flat-file read (~1 min/epoch).
 2. PointPillars builds pillars on-the-fly per frame (plane fit + point sorting every epoch) — took far longer than the precomputed-feature runs
