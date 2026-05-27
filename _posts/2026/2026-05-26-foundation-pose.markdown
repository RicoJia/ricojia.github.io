---
layout: post
title: "[ML] FoundationPose (CVPR 2024)"
date: 2026-05-26 13:19
subtitle: ""
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine-Learning
---

## Introduction

[Code](https://github.com/NVlabs/FoundationPose), [paper](https://arxiv.org/pdf/2312.08344)

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://pic1.zhimg.com/v2-78c7eeb966c767bdb65337eee9c76a6a_1440w.jpg" height="600" alt=""/>
    </figure>
</p>
</div>

1. To reduce manual effort for large-scale training, FoundationPose introduces a synthetic data generation pipeline built on 3D model databases (GSO, Objaverse), large language models, and diffusion models (Sec. 3.1).
2. For pose estimation, FoundationPose  first initializes global poses uniformly around the object, then refines them with a refinement network. Finally, it forwards the refined poses to a pose selection module that predicts scores, and selects the highest-scoring pose as output.
3. FoundationPose also supports a model-free mode: with a small set of reference images, it uses an object-centric neural field (Sec. 3.2) for novel-view RGB-D rendering in a render-and-compare pipeline. (Skipped in this article because we do not need it.)

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/FHXYVqCs/mermaid-diagram.png" height="700" alt=""/>
    </figure>
</p>
</div>

## Step 1: Data Generation for Pose-Refinement and Scoring Networks

A foundation model is a large, general-purpose model trained on a very large and diverse dataset, so it learns reusable 3D priors that transfer to many new objects and tasks.

FoundationPose is trained mostly on synthetic data. The authors use large 3D asset databases, including Objaverse and Google Scanned Objects, then augment object appearances using LLM-aided texture prompts and diffusion-based texture synthesis:

1. The system collects object assets from 3D model libraries, including more than 40,000 objects from Objaverse-LVIS and Google Scanned Objects. It covers 1,156 LVIS categories, and each object has a category label such as cup, bottle, or box.
2. These category labels are then given to ChatGPT, allowing the large language model to automatically generate more specific appearance descriptions, such as `"a green ceramic cup with cartoon patterns on its surface."`.
3. These text prompts are then passed to TexFusion (a diffusion model / texture generation model), which generates more realistic and diverse textures for the original 3D objects.
    - TexFusion also receives a randomly initialized noisy texture.
4. Finally, the system uses a physics simulation and rendering engine (Isaac Sim) to randomly generate RGB-D images under different lighting conditions, viewpoints, backgrounds, materials, and occlusions. At the same time, it automatically obtains accurate ground-truth annotations, including 6D pose, depth maps, segmentation masks, and 2D bounding boxes.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/3RDJ2xB7/Screenshot-from-2026-05-26-12-22-23.png" height="300" alt=""/>
    </figure>
</p>
</div>

## Step 2: 2D Object Detection Input

The pose estimation pipeline takes the following inputs:

- RGB image
- Depth image
- 2D bounding box or mask of the target object
- CAD/mesh or reference views

## Step 3: Pose Initialization

FoundationPose starts with many rough pose guesses.

1. For translation, they use the median depth inside the **detected 2D bounding box** to estimate an initial 3D position.
2. For rotation, they uniformly sample viewpoints from an icosphere around the object and add discretized in-plane rotations.

```python
def initialize_global_poses(bbox, depth):
    center_3d = median_depth_point_inside_bbox(bbox, depth)

    rotations = []
    for viewpoint in sample_icosphere_viewpoints():
        for roll in discrete_inplane_rotations():
            rotations.append(compose(viewpoint, roll))

    poses = []
    for R in rotations:
        poses.append(Pose(R=R, t=center_3d))

    return poses
```

## Step 4: Render-and-Compare Refinement

For each coarse pose hypothesis:

1. Given a guessed pose, we project the hypothesized object center into the image and use that projected point as the crop center.
2. Estimate the projected object diameter.
3. Then we crop the observed RGB-D image around this pose-conditioned center.
    - If the pose hypothesis has the wrong translation, the real object will appear shifted inside the observed crop relative to the rendered object. This shift gives the refinement network a clear signal for how to update the translation.
4. Feed rendered RGB-D and observed RGB-D into a neural refiner.
5. Predict a small pose update.

```python
def refine_pose(pose, rgb, depth, object_representation):
    rendered = render(object_representation, pose)
    crop_center = project_object_origin(pose)
    crop_size = project_object_diameter(pose, object_diameter)
    observed_crop = crop(rgb, depth, center=crop_center, size=crop_size)

    delta_R, delta_t = refinement_network(rendered, observed_crop)

    new_pose = apply_pose_update(pose, delta_R, delta_t)
    return new_pose
```

## Step 5: Refinement Network Learns Features

The CNN extracts local visual alignment cues. The transformer lets different regions of the rendered and observed inputs compare with each other and form richer feature vectors. The refiner has two RGB-D branches:

```
rendered RGB-D branch     observed RGB-D branch
          ↓                         ↓
      shared CNN encoder       shared CNN encoder
          ↓                         ↓
        feature maps concatenated
          ↓
        residual CNN blocks
          ↓
        patch/token representation
          ↓
        transformer encoder
          ↓
   predict translation update + rotation update
```

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://pic4.zhimg.com/v2-522b2f3a9aea5da8d1002e085c858101_1440w.jpg" height="300" alt=""/>
    </figure>
</p>
</div>

The main method for pose generation (also used in GPD and GraspNet) is roughly: "generate *a bunch of* pose hypotheses, then find the one that best matches the observation." Similar to particle filtering, the main limitation is the number of hypotheses. The `coarse + refine` strategy samples only a small set of poses, then predicts refinement deltas. This reduces raw sampling effort.  

Instead of predicting one monolithic SE(3) transform, FoundationPose predicts rotation and translation updates separately in the camera frame.

```
delta_t_cam, delta_R_cam = network(rendered, observed)

t_new = t_old + delta_t_cam
R_new = delta_R_cam @ R_old
```

## Step 6: Pose Selection / Ranking

After refinement, FoundationPose may still have many candidate poses. It needs to choose the best one. The pose selection module uses a hierarchical ranking network:

1. Input $K$ poses.

2. For each pose hypothesis:
   - render the object at that pose;
   - crop the observed RGB-D image using pose-conditioned cropping;
   - encode the `(rendered RGB-D crop with the observed RGB-D crop)` pair into a feature embedding.

3. First-level self-attention comparison: render-vs-observation comparison.
   - For each candidate pose, the network asks: "If this pose were correct, would the rendered object look aligned with the real observation?"
   - This produces one alignment-quality vector per pose.

4. Second-level self-attention comparison: pose-vs-pose comparison. The $K$ pose embeddings are treated as a sequence and passed through multi-head self-attention.

    - This allows each pose's score to be computed relative to the other candidates, instead of assigning an isolated absolute score to each pose independently.
    - This is analogous to "Do not grade each student in isolation. Let the students be compared against the whole class, so the best explanation stands out."

5. The attended embeddings are linearly projected to $K$ scalar scores.

6. The pose with the highest score is selected as the final pose.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://pica.zhimg.com/v2-4058303bff38ffeadff7017d41f94f8a_1440w.jpg" height="300" alt=""/>
    </figure>
</p>
</div>

```python
def select_best_pose(poses, rgb, depth, object_representation):
    embeddings = []

    for pose in poses:
        rendered = render(object_representation, pose)
        observed_crop = pose_conditioned_crop(rgb, depth, pose)
        e = pair_encoder(rendered, observed_crop)
        embeddings.append(e)

    # Compare all hypotheses jointly
    contextual_embeddings = self_attention_over_hypotheses(embeddings)

    scores = linear_score_head(contextual_embeddings)

    return poses[argmax(scores)]
```

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://pica.zhimg.com/v2-a19b3afbcb1ce6f35feccf1869e24db4_1440w.jpg" height="300" alt=""/>
    </figure>
</p>
</div>

## Step 7: Tracking Workflow

For tracking, FoundationPose does not need to globally sample many poses every frame. Once the pose in frame $t-1$ is known, the pose in frame $t$ is probably nearby. Use the previous pose as the initial hypothesis and refine it.

```python
# instead of proposing many pose hypotheses
rendered = render(object_representation, last_pose)
observed_crop = pose_conditioned_crop(rgb_t, depth_t, last_pose)
delta_R, delta_t = refinement_network(rendered, observed_crop)
pose_t = apply_pose_update(last_pose, delta_R, delta_t)
```

This is why tracking is much faster than single-frame pose estimation. The paper says it extracts feature maps from two RGB-D input branches using a shared CNN encoder, concatenates the features, tokenizes them into patches with positional embeddings, then uses transformer encoders to predict translation and rotation updates.

---

## Step 8 — Loss: Contrast Validation for Pose Ranking

Let's score some candidate poses! During training, ground truth is available, so we can decide which candidate pose is better.

First, define the **ADD metric**. For predicted pose $T = (R, t)$ and ground-truth pose $T^* = (R^*, t^*)$, ADD measures the average distance between object model points transformed by the predicted and ground-truth poses:

$$
\mathrm{ADD}(T, T^*) =
\frac{1}{|\mathcal{M}|}
\sum_{x \in \mathcal{M}}
\left\| Rx + t - (R^*x + t^*) \right\|_2
$$

where:

- $x$: a sampled 3D point on the CAD/object model.
- $\mathcal{M}$: the set of sampled object points.
- Smaller ADD means the pose is closer to ground truth.

Then, let's define **positive and negative pose samples**. Given two candidate poses $T_i$ and $T_j$:

```python
if ADD(T_i, T_gt) < ADD(T_j, T_gt):
    positive = T_i
    negative = T_j
else:
    positive = T_j
    negative = T_i
```

Putting the above together, we can define **Pose-Conditioned Ranking Loss**:

$$
\mathcal{L} = \max(0, s_{neg} - s_{pos} + m)
$$

where:

- $s_{pos}$: score of the better pose.
- $s_{neg}$: score of the worse pose.
- $m$: margin.

Desired condition:

$$
s_{pos} \ge s_{neg} + m
$$

### 7-1 Example of Loss Calculation

Example with $m = 0.2$:

If the ADD metric is:

```
Pose A ADD = 2 cm  -> positive
Pose B ADD = 8 cm  -> negative
```

A well-trained network should output `score(A) > score(B)`.

If we see:

```text
score(A) = 0.7
score(B) = 0.4
```

$$
\mathcal{L} = \max(0, 0.4 - 0.7 + 0.2) = 0
$$

Good: positive pose is ahead by enough.

But:

```text
score(A) = 0.55
score(B) = 0.50
```

$$
\mathcal{L} = \max(0, 0.50 - 0.55 + 0.2) = 0.15
$$

Bad: A is better, but not scored sufficiently higher.

---

### 7-2 Positive Poses with Rotations That Are Too Far Are Not Included in Loss

FoundationPose only uses pairs where the positive pose is close enough to the ground-truth viewpoint in [geodesic distance](https://ricojia.github.io/2017/01/23/math-distance-metrics/):

$$
d_{geo}(R_{pos}, R^*) < \theta
$$

where:

- $R_{pos}$: rotation of the positive pose.
- $R^*$: ground-truth rotation.
- $\theta$: predefined threshold.

This filtering excludes pose pairs in which the better pose is still too far rotated from ground truth, because in those pairs both estimates are "bad." They are more likely to introduce noise during training.

### 7-3 Summary of Loss Calculation

```python
loss = 0

for i in range(K):
    for j in range(i + 1, K):
        add_i = ADD(T_i, T_gt)
        add_j = ADD(T_j, T_gt)

        if add_i < add_j:
            pos, neg = T_i, T_j
            s_pos, s_neg = score_i, score_j
        else:
            pos, neg = T_j, T_i
            s_pos, s_neg = score_j, score_i

        if geodesic_distance(pos.R, T_gt.R) < theta:
            loss += max(0, s_neg - s_pos + margin)
```

## Performance

The paper reports single-object pose estimation taking about 1.3 seconds, while tracking runs at 32 Hz because it only needs refinement and not many global hypotheses.

[Nvidia claims that](https://nvidia-isaac-ros.github.io/concepts/pose_estimation/foundationpose/index.html?utm_source=chatgpt.com) FoundationPose tracking, which runs at over 120 FPS on Jetson Thor.

The paper also states that it could handle moderate amount of occlusion well. This is probably because the network learns the pose - RGBD alignment of objects instead of fix shape templates.

## Validation

FoundationPose’s official framing is: [it can be applied to a novel object at test time without fine-tuning if you provide either a CAD model or a small number of reference images](https://github.com/NVlabs/FoundationPose). FoundationPose’s model-based mode expects a CAD model or mesh file of the target object because it needs to render the object from candidate poses. In practice, you usually do not need a fully parametric SolidWorks-style CAD model; [a decent renderable mesh is often enough](https://deepwiki.com/NVlabs/FoundationPose/4.1-model-based-pose-estimation).

Here is a practical guide for [FoundationPose usage](https://deepwiki.com/NVlabs/FoundationPose/4.1-model-based-pose-estimation).

### Metric - AR

Average Recall (AR) is often used in the BOP benchmark style 6D pose estimation. Here we define a correct pose being `pose is correct if error < threshold`, then `recall = number of correctly estimated poses / number of ground-truth poses`. Here we have 3 BOP pose errors:

1. VSD — Visible Surface Discrepancy. Measures whether the visible rendered surface of the predicted pose matches the visible surface of the ground truth. Good for occlusion because it focuses on what is visible.

2. MSSD — Maximum Symmetry-Aware Surface Distance: measures 3D surface mismatch while accounting for object symmetries. BOP describes MSSD as a surface-deviation metric that considers predefined global object symmetries.

3. MSPD — Maximum Symmetry-Aware Projection Distance: Measures 2D projection mismatch while also considering object symmetries. BOP describes MSPD as a symmetry-aware projection-distance metric.

### Data Synthesis: TexFusion Is Experimental

TexFusion’s method takes a text prompt plus mesh geometry and produces a UV-parameterized texture using Stable Diffusion as the text-to-image backbone. Its core idea is multi-view diffusion sampling, aggregating views through a latent texture map, then fusing decoded RGB views into a texture map.

Caveat: the NVIDIA TexFusion project page describes the method, but I do not see an official NVIDIA code release on that page. [The GitHub repo I found is explicitly an unofficial](https://github.com/silence401/Texfusion.git) implementation and mentions open issues with “vgg loss” and “quality,” so treat it as experimental.

## Model Specs

- Model size is likely <50M. I didn't find a formal published number on the model size, the refine model and score model are small CNN encoders plus 512-dim transformer/multi-head-attention layers, with default RGB-D input c_in=6, batch norm enabled, and axis-angle rotation output

- License: [NVidia license](https://github.com/NVlabs/FoundationPose/blob/main/LICENSE), for research / evaluation. However, [the NVidia NGC version](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/foundationpose?utm_source=chatgpt.com&version=1.0.1_onnx) is potentially commercially usable

## Appendices

- [Objaverse-XL](https://objaverse.allenai.org/): "A Universe of 10M+ 3D Objects". (Zero123-XL transforms single image into 3D model using Dreamfusion. )

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://objaverse.allenai.org/objaverse-xl/zero123-xl.jpg" height="300" alt=""/>
    </figure>
</p>
</div>

- [GSO](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/): 1032 3D importable objects for Gazebo
- [BOP Benchmark for 6D Object Pose Estimation](https://bop.felk.cvut.cz/leaderboards/pose-estimation-unseen-bop23/core-datasets/)
