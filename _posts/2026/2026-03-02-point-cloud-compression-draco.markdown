---
layout: post
title: "[Point Cloud Compression] Draco"
date: 2026-03-02 13:19
subtitle:
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine-Learning
---
## Introduction

[Draco](https://github.com/google/draco) is Google's open-source library for compressing 3D geometric meshes and point clouds. It quantizes floating-point attributes (positions, normals, colors, texture coordinates) into compact integer representations, then exploits spatial redundancy through prediction and entropy coding. For meshes it also compresses connectivity (the triangle list); for point clouds it skips connectivity and focuses on the points.

### Pipeline Overview

1. **Quantize attributes** — map floats onto an integer grid. This is the only lossy step; the user controls precision via the number of quantization bits.
2. **Predict from neighbors** — instead of storing raw quantized values, predict each point from previously decoded neighbors and store only the small residual (actual − predicted).
3. **Entropy-code the residuals** — feed the residual stream into an rANS (range Asymmetric Numeral Systems) encoder that builds **empirical frequency tables** from the data itself (no learned CDF).

### Is Draco Lossy or Lossless?

Quantization is lossy — floats are irreversibly rounded to integers. After quantization, however, prediction + entropy coding are fully lossless. Mesh connectivity encoding is also lossless. So the user controls the trade-off: more quantization bits → higher precision but larger output.

## Example

### Step 1 — Quantization

Suppose your input positions are:

```python
points = [
    [0.12, 0.18, 0.05],
    [0.15, 0.20, 0.05],
    [0.18, 0.21, 0.04],
    [0.80, 0.90, 0.10],
]
```

**Quantization bits** determine grid resolution. With $n$ bits there are $2^n$ levels. For example, if the bounding box along one axis spans $[0, 1]$ and we use 8 bits, the step size is $1 / (2^8 - 1) \approx 0.0039$. Fewer bits → smaller file but coarser precision.

Here, imagine a simple step size of `0.01`:

```python
quantized = [
    [12, 18, 5],
    [15, 20, 5],
    [18, 21, 4],
    [80, 90, 10],
]
```

Draco stores the **bounding-box minimum** and the **step size** so the decoder can reconstruct approximate floats: $\text{float} = \text{min} + \text{quantized\_int} \times \text{step}$.

### Step 2 — Prediction and Residual Coding

Imagine a toy predictor: "predict the next point from the previous one." Instead of storing every quantized point directly, we store residuals:

```python
stored = [
    [12, 18, 5],    # first point stored as-is
    [ 3,  2, 0],    # [15,20,5] - [12,18,5]
    [ 3,  1,-1],    # [18,21,4] - [15,20,5]
    [62, 69, 6],    # [80,90,10] - [18,21,4]
]
```

The residuals are still `(x, y, z)` integers — the dimensionality doesn't shrink. The win is that nearby points produce **small values concentrated around zero**, which have much lower entropy than the raw coordinates. **Low-entropy integers compress far better under entropy coding.** Draco uses more sophisticated prediction schemes than this toy example (see below), but the core idea is the same.

### Step 3 — Entropy Coding

Draco feeds the residual stream into an **rANS** encoder. rANS builds a frequency table from the actual symbol distribution in the current data — there is no pre-trained or learned CDF. Because the residuals are small and clustered, the frequency table is sharply peaked and the resulting bitstream is very compact.

## KD-Tree Ordering (Point Clouds)

For point clouds, Draco uses a KD-tree to reorder points so that spatially close points are consecutive in the encoding stream. This makes the residuals from prediction much smaller.

The KD-tree works by:

- Recursively splitting points along alternating coordinate axes
- Keeping spatially nearby points clustered in the traversal order
- Encoding splits and local structure compactly

Using the same example points (projected to 2D for clarity):

```python
points = [
    [12, 18],
    [15, 20],
    [18, 21],
    [80, 90],
]
```

A bad arbitrary order might be:

```python
bad_order = [
    [12, 18],
    [80, 90],
    [15, 20],
    [18, 21],
]
```

That jumps across space, producing large residuals. A KD-tree split on `x` separates the cluster `{[12,18], [15,20], [18,21]}` from `{[80,90]}`, so the traversal order keeps **potentially** nearby points together:

```python
kd_order = [
    [12, 18],
    [15, 20],
    [18, 21],
    [80, 90],
]
```

Now consecutive points are spatially close, and prediction residuals stay small.

## Prediction Schemes

Draco uses different prediction strategies depending on the data type:

- **Point clouds (KD-tree encoder)**: predict each point from the center of its KD-tree cell or from previously decoded neighbors in the same cell.
- **Meshes (parallelogram prediction)**: given a triangle that shares an edge with an already-decoded triangle, predict the new vertex by mirroring the opposite vertex across the shared edge. This exploits the local regularity of mesh surfaces.

## Mesh Connectivity Compression

For a **mesh**, Draco also compresses the triangle list:

```python
vertices = [
    [0.0, 0.0, 0.0],  # v0
    [1.0, 0.0, 0.0],  # v1
    [1.0, 1.0, 0.0],  # v2
    [0.0, 1.0, 0.0],  # v3
]

faces = [
    [0, 1, 2],
    [0, 2, 3],
]
```

A plain format stores the two triangles as raw index triples. Draco instead uses an edge-breaker or similar traversal-based connectivity encoding that exploits shared edges and adjacency patterns, encoding connectivity far more compactly than a raw triangle list.

## Summary

**Draco works by quantizing geometry, predicting values from neighboring structure, storing only compact residuals, and entropy-coding both attributes and mesh connectivity.**
