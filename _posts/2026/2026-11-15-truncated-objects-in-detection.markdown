---
layout: post
title: "[Robotics] Handling Truncated Objects In Detection"
date: 2026-11-15 13:19
subtitle: Truncation Ratio, Truncation Augmentation, Separate Metrics
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - robotics
---

A truncated object is one that is only partially inside the sensor's field of view. Its crop doesn't look like a fully seen object, so it can confuse both training and evaluation.

## How Public Datasets Handle Truncated Data

Set a truncation threshold. If an object is truncated too much, the sample does not count as positive or negative, in training or evaluation:

- **KITTI** labels each object with a truncation ratio from 0 (fully visible) to 1 (fully outside the image). Each difficulty level has a maximum truncation (0.15 for Easy, 0.3 for Moderate, 0.5 for Hard), and objects above it are ignored rather than counted as misses.
- **nuScenes** and **Waymo** do the same with point counts: boxes with no LiDAR points inside are dropped from evaluation, and Waymo puts boxes with few points into a harder difficulty level.

## Can Surface Area Detect Truncation?

To find truncated objects in our training data, first find the full detections: boxes at least 0.3 m from the edge of the field of view. The 0.3 m is my estimate of how much a labeled box could "miss".

If we can get an idea of the surface area of full detections, and if that area is consistent per object, maybe we can use it as a better criterion for truncation detection: a crop with much less area than usual is probably truncated.

**Finding: the area is not consistent.** For boxes at least 1 m from the edge, the median area is 1.27 m², but the middle 80% spans 0.79–1.86 m². That spread is too wide to separate a truncated object from a small full one.

## Coping Mechanisms

### Training

- **Truncation augmentation:** fully seen objects are cut randomly during training, so the model sees truncated examples of every class.
- **Truncated mines become "unknown"**, which is skipped during training, just like KITTI's ignored objects.

### Evaluation

Recall and FPR are reported separately for full and truncated crops. We did find that recall and FPR both increase as the distance to the edge increases.

### Runtime

- The track classification remains "object, unknown" until the object gets a good view.
- Downweight detections with edge beam count > 0, i.e., detections that touch the edge of the field of view.

### Open Idea

Maybe there's a way for us to merge partial observations of the same track into a whole object?
