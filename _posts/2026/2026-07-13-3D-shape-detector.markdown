---
layout: post
title: "[Robotics] 3D-shape-detection-from-point-clouds"
date: 2026-07-13 13:19
subtitle: BEV, Model Training Experience
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

## 2. Model Training Experience

### 2-1 Handcrafted Features Are Good First Pass Mechanism. Later Should be Replaced By A Learned Encoder

We actually tried creating hand-crafted features, then feeding them into the BEV network. It was basically early-stage work that helped us identify key elements in noise for data synthesis and quickly bring up a semi-working model:

- Hand-crafted 8-channel features (p90 height, relief, shadow density, etc.) are interpretable
- Hand-crafted features get precomputed once into features.bin, so training is a flat-file read (~1 min/epoch).

However later we realized that the model has plateaued. Switching to af learned CNN encoder helped.

### 2-2. Downsampling & Upsampling Can Learn Spatial Correlation

Real sonar isn't equally noisy everywhere. Long range is noisier than short. Grazing angles are noisier than perpendicular. Dark regions are noisier than bright. So the noise level must vary across the image in a spatially correlated way

This didn't work well:

```python
self.head = nn.Conv2d(width, 4, 1)     # 1x1 conv, full 128x128 resolution
```

**A 1×1 convolution at full resolution means every pixel's amplitude is set independently.** Nothing couples neighbours. So the network can — and does — make one pixel 100× louder than the one beside it

One remedy is to predict $A$ at low resolution, then upsample.

```python
#amplitude/gain describe sensor geometry -> low-frequency by construction
coarse = F.adaptive_avg_pool2d(features, 8)          # 128x128 -> 8x8
params = self.head(coarse)                            # 1x1 conv on 8x8
params = F.interpolate(params, size=(128, 128),
                       mode="bilinear", align_corners=False)
```

With an 8×8 grid upsampled bilinearly, $A$ can only vary on a ~16-pixel scale. A single-pixel amplitude spike is impossible.
