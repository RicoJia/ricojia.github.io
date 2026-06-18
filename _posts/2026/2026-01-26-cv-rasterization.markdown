---
layout: post
title: "[CV] Rasterization"
date: 2026-01-26 13:19
subtitle: Barycentric Coordinates
comments: true
header-img: img/post-bg-alibaba.jpg
tags:
  - CV
---

## Goal of Rasterization: Project 3D Triangles Down to 2D Triangles

Let's pretend a mini 3D mesh has two triangles:

```python
faces = [
    [0, 1, 2],  # triangle 1
    [3, 4, 5],  # triangle 2
]
```

We project these two triangles onto a tiny **5×5 image** with **two projected triangles**.

```text
pixel grid, x →
y

0   . . . . .
1   . A A . .
2   . A B B .
3   . . B B .
4   . . . . .
```

Where:

```text
A = triangle 1 is visible
B = triangle 2 is visible
. = background
```

Rasterization's goal is to turn that into a `triangle_id` map:

```text
triangle_id map

0 0 0 0 0
0 1 1 0 0
0 1 2 2 0
0 0 2 2 0
0 0 0 0 0
```

Let's store triangle id is stored in the **4th channel** of `rast`. So in its output:

```python
rast[1, 1, 3] = 1  # pixel sees triangle 1
rast[2, 2, 3] = 2  # pixel sees triangle 2
rast[0, 0, 3] = 0  # background
```

But rasterization stores more than triangle id. It also stores **barycentric coordinatesof the 2D triangle**: `rast[y, x] = [u, v, w, triangle_id]`. For one pixel on triangle 1: barycentric = [0.2, 0.3, 0.5]. For one pixel on triangle 2: barycentric = [0.1, 0.7, 0.2]. Note that  is `w = 1 - u - v`, see barycentric coordinates as below

---

## How to get 3D coordinates of a 2D point within 2D Triangle? Interpolate XYZ

Suppose triangle 1 has these 3D camera-frame vertices:

```python
V0 = [0.0, 0.0, 1.0]
V1 = [1.0, 0.0, 1.0]
V2 = [0.0, 1.0, 1.0]
```

And triangle 2 has:

```python
V3 = [1.0, 0.0, 0.8]
V4 = [2.0, 0.0, 0.8]
V5 = [1.0, 1.0, 0.8]
```

At pixel `(1, 1)`, rasterization says: `rast[1,1] = [0.2, 0.3, 0.5, 1]`. So `dr.interpolate(V_cam, rast, faces)` does:

```python
xyz[1,1] = 0.2 * V0 + 0.3 * V1 + 0.5 * V2 = [0.3, 0.5, 1.0]
xyz[2,2] = 0.1 * V3 + 0.7 * V4 + 0.2 * V5 = [1.7, 0.2, 0.8]
```

### Same thing for color

Suppose triangle 1 is red-ish at its vertices:

```python
C0 = [1.0, 0.0, 0.0]
C1 = [1.0, 0.2, 0.0]
C2 = [1.0, 0.0, 0.2]
```

At pixel `(1, 1)`:

```python
color = 0.2 * C0 + 0.3 * C1 + 0.5 * C2 = [1.0, 0.06, 0.10]
```

---

## What if two triangles overlap?

Now suppose both triangles cover pixel `(2, 2)`.

```text
triangle 1 depth at pixel = 1.0
triangle 2 depth at pixel = 0.8
```

The farther triangle is ignored for that pixel. The **z-buffer** buffers the depth of each pixel

---

## Tiny complete pseudocode

```python
for each pixel:
    best_depth = infinity
    best_triangle = 0
    best_bary = None

for tri_id, (i0, i1, i2) in enumerate(faces):
    p0, p1, p2 = project(vertices[i0], vertices[i1], vertices[i2])

    for pixel in pixels_inside_triangle(p0, p1, p2):
        u, v, w = barycentric(pixel, p0, p1, p2)

        depth = u * p0.z + v * p1.z + w * p2.z

        if depth < best_depth[pixel]:
            best_depth[pixel] = depth
            best_triangle[pixel] = tri_id
            best_bary[pixel] = [u, v, w]

# Later:
for each pixel:
    tri_id = best_triangle[pixel]

    if tri_id == 0:
        continue

    i0, i1, i2 = faces[tri_id]

    u, v, w = best_bary[pixel]

    xyz[pixel] = (
        u * vertices_cam[i0]
      + v * vertices_cam[i1]
      + w * vertices_cam[i2]
    )
```

So with two triangles, rasterization is basically building this table:

```text
pixel     visible triangle     barycentric coords     depth
(1, 1)    triangle 1            [0.2, 0.3, 0.5]       1.0
(2, 2)    triangle 2            [0.1, 0.7, 0.2]       0.8
(3, 2)    triangle 2            [0.4, 0.4, 0.2]       0.8
```

Then interpolation uses that table to create rendered RGB, XYZ, depth, and normal maps.

## Barycentric Coordinates

```text
        p2
       /\
      /  \
     / p  \
    /______\
  p0        p1

O
```

To express point `p` from the origin, start from `p0` and move along the two triangle edges:

```text
p = p0 + α(p1 - p0) + β(p2 - p0)
```

Rearranging:

```text
p = (1 - α - β)p0 + αp1 + βp2
```

So any point inside the triangle can be written as a weighted sum of the three vertices:

```text
p = u*p0 + vp1 + (1-u-v) * p2
```
