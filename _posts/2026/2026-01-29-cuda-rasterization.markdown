---
layout: post
title: "[CUDA] rasterization"
date: 2026-01-29 13:19
subtitle: , nvdiffrast, kornia
comments: true
header-img: img/post-bg-alibaba.jpg
tags:
  - CUDA
---

## Nvdiffrast

In FoundationPose-style tracking, `nvdiffrast + kornia` are the **GPU crop/render pair builder** before the pose networks. The pose network does not directly look at the whole camera frame. Instead, for each candidate pose, the code builds a pair:

```text
A = synthetic CAD render at candidate pose
B = real RGB-D crop from camera at same projected window

RefineNet(A, B) -> delta rotation + delta translation
ScoreNet(A, B)  -> how good this pose hypothesis is
```

For the **scorer**, it is in `learning/training/predict_score.py::make_crop_data_batch`. It does the same kind of render-and-warp setup, then `ScoreNet` compares the rendered hypothesis against the real observation

- nvdiffrast = "what should the CAD object look like if this pose is true?" renders the CAD model. `nvdiffrast_render` and outputs crop transforms
  - `nvdiffrast` is basically a **fast CAD renderer**, which is foundamentally a low-level differentiable rasterization library. NVIDIA describes it as exposing the graphics pipeline pieces: rasterization, interpolation, texturing, and antialiasing, all GPU accelerated with CUDA. It does **not** give you camera models or pose logic for free; FoundationPose supplies those. ([NV Labs][4])
  - Its rasterization output is `rast[y, x] = [u, v, z/w, triangle_id]`
  - The actual calls are `dr.rasterize`, `dr.interpolate`, and optionally `dr.texture`. ([GitHub][3])

- kornia = "crop/warp the real camera image to the same coordinate frame" `kornia.geometry.transform.warp_perspective` to crop/warp real RGB. optionally with rendered RGB, XYZ maps, and optionally normals.

- network    = "compare rendered candidate vs real observation"

nvdiffrast:

```python
def nvdiffrast_render_pose(mesh, T_obj_cam, K, H, W):
    # mesh vertices in object frame
    V_obj = mesh.vertices          # [N, 3]
    F = mesh.faces                 # [M, 3]

    # 1. Transform object vertices into camera frame
    V_cam = transform(T_obj_cam, V_obj)

    # 2. Project to clip space / screen space
    P = projection_matrix_from_intrinsics(K, H, W)
    V_clip = P @ opencv_to_opengl @ T_obj_cam @ homo(V_obj)

    # 3. Rasterize triangles
    rast = dr.rasterize(glctx, V_clip, F, resolution=(H, W))

    # 4. Interpolate useful per-pixel attributes
    xyz_map   = dr.interpolate(V_cam, rast, F)
    depth_map = xyz_map[..., 2]
    color     = dr.interpolate(vertex_color_or_texture, rast, F)
    normal    = dr.interpolate(vertex_normals_cam, rast, F)

    return color, depth_map, normal, xyz_map
```

- nvdiffrast’s rasterizer takes clip-space vertex positions and triangle indices; the output is a 4-channel map per pixel: roughly (u, v, z/w, triangle_id). Pixels with no triangle get zeros. u and v are barycentric coordinates inside the selected triangle, z/w is normalized depth, and triangle_id tells which triangle won at that pixel.
  - The way it does it is: magine your phone CAD mesh is made of many tiny triangles.  After projection, each 3D triangle becomes a 2D triangle on the image plane. Rasterization loops over image pixels and asks: Does this pixel center fall inside this projected triangle?
        1. If yes, is this triangle closer than the previous triangle at this pixel?
        2. If yes, store:
            - triangle_id
            - barycentric coordinates
            - depth

## Warping

Conceptually, a warp is inverse sampling. For each output crop pixel, find where to sample in the source image.
Source image

```
        p0 ______ p1
          /      /
         /      /
      p3/______/p2

```

But we need output crop

```
q0 ______ q1
  |      |
  |      |
q3|______|q2
```

The warp maps output square corners -> source quadrilateral corners

1. Take output pixel coordinate (x_out, y_out)
2. Use homography H to find source coordinate (x_src, y_src)

    ```
    [ x' ]   [ h00 h01 h02 ] [ x_out ]
    [ y' ] = [ h10 h11 h12 ] [ y_out ]
    [ z' ]   [ h20 h21 h22 ] [   1   ]

    x_src = x' / z'
    y_src = y' / z'
    ```

    - The divide by z' is what allows perspective effects.
3. Sample source image at that coordinate
    - Usually (x_src, y_src) is not an exact integer pixel, e.g., there is no exact pixel at (2.3, 1.7), so the warp samples nearby pixels. So people usually sample using billinear interpolation:

        ```
        source pixels around (2.3, 1.7):

        (2,1), (3,1)
        (2,2), (3,2)

        weighted average them
        ```

4. Write sampled value into output pixel

## What `kornia` does

`kornia` is used as a PyTorch/GPU image geometry library. One of its important operators is `warp_perspective`, which applies a 3x3 perspective transform to a tensor image and returns a warped `(B, C, H, W)` tensor. ([kornia.readthedocs.io][5])

In FoundationPose, the code first computes a crop transform:

```python
tf_to_crops = compute_crop_window_tf_batch(
    pts=mesh.vertices,
    H=H,
    W=W,
    poses=ob_in_cams,
    K=K,
    crop_ratio=crop_ratio,
    out_size=(crop_W, crop_H),
    method="box_3d",
)
```

Then Kornia applies that transform to the real camera image and real XYZ map:

```python
rgbB = kornia.geometry.transform.warp_perspective(
    real_rgb[None].expand(B, -1, -1, -1),
    tf_to_crops,
    dsize=render_size,
    mode="bilinear",
)

xyzB = kornia.geometry.transform.warp_perspective(
    real_xyz_map[None].expand(B, -1, -1, -1),
    tf_to_crops,
    dsize=render_size,
    mode="nearest",
)
```

So `kornia` makes the “B” side:

```text
real RGB crop
real XYZ/depth crop
optional real normal crop
```

The refiner code then concatenates these:

```python
A = torch.cat([pose_data.rgbAs, pose_data.xyz_mapAs], dim=1).float()
B = torch.cat([pose_data.rgbBs, pose_data.xyz_mapBs], dim=1).float()

output = RefineNet(A, B)
```

That exact pattern appears in the refiner: crop data is built, `A` and `B` are concatenated, the model predicts pose deltas, and then the pose is updated. ([GitHub][1])

## What “graphing” means here

“Graphing” here should mean **CUDA Graph capture**, not a neural-network graph and not graph neural networks.

Normally, every frame runs many small GPU operations:

```text
render kernels
interpolation kernels
warp kernels
concat kernels
normalization kernels
CNN kernels
pose-update kernels
```

Each kernel launch has CPU overhead. CUDA Graphs record a fixed sequence of CUDA work once, then replay that same sequence repeatedly. PyTorch’s `torch.cuda.graph` is specifically a context manager that captures CUDA work into a `CUDAGraph` for later replay. ([PyTorch Documentation][6])

The catch: CUDA graphs want the **same operation sequence, same tensor shapes, and same memory addresses** on replay. NVIDIA’s PyTorch CUDA Graph guidance states that current PyTorch CUDA graphs require static execution patterns: same sequence of operations and same memory addresses every replay. ([NVIDIA Docs][7])

So for your tracking path, graphing means:

```text
allocate fixed input buffers once
capture:
    nvdiffrast render
    kornia warps
    concat A/B
    RefineNet forward
    pose update
replay every frame:
    copy new RGB/depth/pose into same buffers
    graph.replay()
    read output pose
```

[1]: https://github.com/NVlabs/FoundationPose/blob/main/learning/training/predict_pose_refine.py "FoundationPose/learning/training/predict_pose_refine.py at main · NVlabs/FoundationPose · GitHub"
[3]: https://github.com/NVlabs/FoundationPose/blob/main/Utils.py "FoundationPose/Utils.py at main · NVlabs/FoundationPose · GitHub"
[4]: https://nvlabs.github.io/nvdiffrast/ "nvdiffrast"
[5]: https://kornia.readthedocs.io/en/latest/geometry.transform.html "kornia.geometry.transform - Kornia"
[6]: https://docs.pytorch.org/docs/2.12/generated/torch.cuda.graphs.graph.html "graph — PyTorch 2.12 documentation"
[7]: https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/handling-dynamic-patterns.html "Handling Dynamic Patterns — CUDA Graph Best Practice for PyTorch"
