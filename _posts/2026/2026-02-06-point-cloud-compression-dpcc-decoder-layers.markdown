---
layout: post
title: "[ML] D-PCC Decoder Layers"
date: 2025-02-06 13:19
subtitle: isohedron, losses
header-img: img/post-bg-o.jpg
tags:
  - Machine-Learning
comments: true
---
---

## Decoder

We need multiple upsampling block, instead of a major one. Like instead of doing upsampling `x3 -> x3 -> x3`, we just do `x27`. Why?

1. upsampling x27 in training might require large gradient changes, which might introduce numerical instabilities
2. this would require input features to capture extremely fine details.
3. receptive field is the same, but more intermediate non-linearity between combinations of input elements is learned.

This is similar to super pixel:

```
2x -> 2x -> 2x
```

Instead of doing it in one shot:

```
->8x
```

### Icosahedron

An **icosahedron** is a regular solid with 20 triangular faces and 12 vertices. In this project, `icosahedron2sphere(level)` uses it to generate nearly uniformly distributed directions on a sphere — these serve as candidate upsampling directions when reconstructing point clouds.

A unit icosahedron has all edges of equal length. This holds if and only if its 12 vertices are:

$$(0, \pm 1, \pm \varphi), \quad (\pm 1, \pm \varphi, 0), \quad (\pm \varphi, 0, \pm 1)$$

where $\varphi$ is the **golden ratio**:

$$\varphi = \frac{1 + \sqrt{5}}{2} \approx 1.618$$

Using any other value would produce unequal edge lengths.

`icosahedron2sphere(level)` works as follows:

1. Project the icosahedron's 12 vertices onto a unit sphere.
2. If `level > 1`, subdivide each triangular face by inserting a new vertex at the midpoint of each edge, then project those new vertices back onto the sphere.
3. Return the resulting directions, which are nearly uniformly distributed over the sphere.

<div style="text-align: center;">
  <div style="display: flex; justify-content: center; align-items: flex-start; gap: 40px; flex-wrap: wrap;">
    <figure>
      <img src="https://i.postimg.cc/Z51RQH4J/image-29.png" height="300" alt="Icosahedron"/>
      <figcaption>Icosahedron (20 faces, 12 vertices)</figcaption>
    </figure>
    <figure>
      <img src="https://i.postimg.cc/L8NX9Stm/icosahedron-sphere-points-level1.png" height="300" alt="Icosahedron Sphere Points Level 1"/>
      <figcaption>Vertices projected onto sphere — Level 1</figcaption>
    </figure>
  </div>
</div>

The 12 base vertices are not perfectly uniform, but each subdivision level makes the distribution increasingly uniform. As the return values of this stage, we return

- vertices as 3D coordinates
- triangles' vertex indices in verticex coordinates above

Below is Level 2 — the midpoints of all edges are added and re-projected:

<div style="text-align: center;">
  <div style="display: flex; justify-content: center; align-items: flex-start; gap: 40px; flex-wrap: wrap;">
    <figure>
      <img src="https://i.postimg.cc/4xBdfG6T/uniform-directions-level2-angular-projection.png" height="300" alt="Uniform Directions Level 2 Angular Projection"/>
      <figcaption>Uniform directions — Level 2 angular projection</figcaption>
    </figure>
    <figure>
      <img src="https://i.postimg.cc/DwBZvhr2/uniform-directions-level2-mesh-quiver.png" height="300" alt="Uniform Directions Level 2 Mesh Quiver"/>
      <figcaption>Uniform directions — Level 2 mesh quiver visualization</figcaption>
    </figure>
  </div>
</div>

### Sub-Pixel / Sub point Convolution

Assume we want to upsample an input by upsample factor `r`

```
H x W x C -> rH x rW x C
```

TODO: Traditionally, this is done by interpolation. However, this tends to yield similar features Why would they duplicate in the first place???
Because these features have small variations, the resultant points are clustered.

**Periodic shuffle is**

```
H x W x C -> H x W x (C*r^2) ---shuffle---> rH x rW x C 
```

You simply interpret the same data in a different shape, like

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/nh6Hs8vF/Chat-GPT-Image-Feb-26-2026-12-24-43-PM.png" height="300" alt=""/>
    </figure>
</p>
</div>

**TODO: Why it works? Channel shuffling + piecing together increase the possiblity of variation of the feature space?? Then this helps gradient descent will optimize parameters with more variation.**

#### Sub-pixel Convolution is similar

Feature Dimension is 4.

```
N x c = 2 x 4
```

wuith `upsampling factor = r`: after convolution:

```
N x c x r
```

Then periodic shuffle:

```
(r * N) x c
```

## Loss

 mean distance, number of upsampled points, Chamfer loss per downsample stage is fed into the loss function, so they are directly penalized:

- Upsampling ratioL: L1(predicted upsample_num  vs  ground truth downsample_num)
- predicted mean distance  vs  true mean distance
