---
layout: post
title: "[Point Cloud Compression] - Sampling"
date: 2026-01-07 13:19
subtitle:
comments: true
tags:
  - CUDA
---
---

## Furthest Point Sampling

Idea: given a set of points

```
P = {P1, P2, ...}
```

Select

```
S = {s1, s2 ...}
```

such that S is maximally spread out. How?

> Add an intial point in P to S. Then iteratively add the point in P whose minimum distance to points in S is maximal

$$
S_{k+1} = arg \space max_{p \in P} \space min_{s \in S} \space |p-s|
$$

### Example

```
P={0,2,3,7,10}
```

We want `M=3` points.

Step 1 - Start with

$$
S_1 = 0
$$

Step 2 - All other points to 0 are

```
D = [0, 4, 9, 49, 100]
```

Pick the farthest `10`

Step 3 - Now `S = [0, 10]`. All other points to these points are

| Point | dist to 0 | dist to 10 | min |
| ----- | --------- | ---------- | --- |
| 0     | 0         | 100        | 0   |
| 2     | 4         | 64         | 4   |
| 3     | 9         | 49         | 9   |
| 7     | 49        | 9          | 9   |
| 10    | 100       | 0          | 0   |
So:

```
D=[0,4,9,9,0]
```

Pick farthest → 3 (or 7)

So the final output is `[0, 3, 10]`

### Usage

FPS is used in  PointNet++ also.  It produces uniform coverage. It reduces redundancy. It keeps geometric structure. It's deterministic unlike random sampling. It's **used in downsampling before neighborhood group**. In ML, it's similar to max-pooling.

FPS's computational complexity is `O(NM)`, where N is the point cloud size, M is the number of points to extract.
