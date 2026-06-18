---
layout: post
title: Computer Vision - Aruco Detection
date: '2026-03-15 13:19'
subtitle: 
comments: true
header-img: img/post-bg-infinity.jpg
tags:
    - Deep Learning
---

## Aruco Detection

Aruco detection has two jobs:

```text
1. Detection: find square marker candidates in the image.
2. Decoding: read the binary ID inside the square.
3. Pose estimation: use the four 2D corners + known marker size to solve camera-to-marker pose.
```

OpenCV’s ArUco module treats markers as square binary fiducial markers; a predefined dictionary has a fixed marker bit size, such as `DICT_6X6_250`, meaning 250 possible markers with a 6×6 binary code area. The dictionary’s minimum Hamming distance controls how well marker IDs can be distinguished and corrected under bit errors. ([OpenCV Documentation][1])

An ArUco marker is basically:

```text
black outer border
binary bit pattern inside
square shape
known physical size
```

Example toy marker with a **4×4 inner code** and a **1-cell black border**:

```text
sampled 6×6 grid

0 0 0 0 0 0
0 1 0 0 1 0
0 0 1 1 0 0
0 1 1 0 0 0
0 0 0 1 1 0
0 0 0 0 0 0
```

Where:

```text
0 = black cell
1 = white cell
```

The detector expects the outer border to be black. Then it reads the inner code:

```text
inner 4×4 code

1 0 0 1
0 1 1 0
1 1 0 0
0 0 1 1
```

The high-level OpenCV-style pipeline is:

```text
RGB image
  -> grayscale + adaptive threshold + find candidate contours
  -> approximate contours as polygons + keep quadrilateral candidates
  -> perspective-warp candidate to square
  -> sample bit grid
  -> rotate marker to find the best rotation
  -> match inner bits to dictionary
  -> return marker ID + 4 corners
```

OpenCV’s ArUco detector works by detecting candidate square regions, warping them into a canonical square form, thresholding the cells, and comparing the resulting bit pattern against the marker dictionary. ([OpenCV Documentation][1])

---

## Step 1 - Finding square candidates

Suppose your camera image contains this marker:

```text
image

       p0 __________ p1
         /          /
        /  marker  /
     p3/__________/p2
```

The detector first thresholds the image:

```text
bright pixels -> white
dark pixels   -> black
```

Then it finds contours and keeps contours that look like quadrilaterals:

```python
for contour in contours:
    polygon = approximate_polygon(contour)

    if len(polygon) != 4:
        reject

    if area_too_small(polygon):
        reject

    if not convex(polygon):
        reject

    keep_as_marker_candidate(polygon)
```

At this point, the detector does **not** yet know the marker ID. It only knows:

```text
this looks like a black-bordered square candidate
```

---

# Step 2 - Perspective warp to a canonical marker image

The candidate in the camera image is tilted:

```text
source image

       p0 __________ p1
         /          /
        /          /
     p3/__________/p2
```

The detector warps it into a clean square:

```text
canonical marker image

q0 __________ q1
 |            |
 |            |
q3|___________|q2
```

This is a homography warp. For each detected image corners

```
p0 = (u0, v0)
p1 = (u1, v1)
p2 = (u2, v2)
p3 = (u3, v3)
```

We choose 4 canonical square points:

q0 = (0, 0)
q1 = (N, 0)
q2 = (N, N)
q3 = (0, N)

We can solve for a 3×3 homography H to sample from the original image:

```text
[ x_q ]   [ h00 h01 h02 ] [ x_p ]
[ y_q ] = [ h10 h11 h12 ] [ y_p ]
[ z_q ]   [ h20 h21 h22 ] [  1  ]

x_p = x_q / z_q
y_p = y_q / z_q
```

This is basically

```python
H = cv2.getPerspectiveTransform(
    src=np.float32([p0, p1, p2, p3]),   # detected image quad
    dst=np.float32([q0, q1, q2, q3]),   # canonical square
)

canonical = cv2.warpPerspective(gray, H, (N, N))
```

After this, the candidate looks front-facing, so it can be divided into cells.

# Step 3 - Sampling the bit grid for Inner Code

For the toy 6×6 marker:

```text
0 0 0 0 0 0
0 1 0 0 1 0
0 0 1 1 0 0
0 1 1 0 0 0
0 0 0 1 1 0
0 0 0 0 0 0
```

The detector samples each cell. A simple version is:

```python
for cell_y in range(6):
    for cell_x in range(6):
        patch = canonical_marker[
            cell_y * cell_size : (cell_y + 1) * cell_size,
            cell_x * cell_size : (cell_x + 1) * cell_size,
        ]

        mean_intensity = patch.mean()

        if mean_intensity > threshold:
            bit = 1
        else:
            bit = 0
```

Then it checks the border:

```text
top row    must be all 0
bottom row must be all 0
left col   must be all 0
right col  must be all 0
```

If the border is not black, reject the candidate.

Then remove the border and keep the inner code:

```text
1 0 0 1
0 1 1 0
1 1 0 0
0 0 1 1
```

---

# Step 4 - Rotation Correction

The camera may see the marker rotated:

```text
original code

1 0 0 1
0 1 1 0
1 1 0 0
0 0 1 1
```

Rotated 90 degrees:

```text
0 1 0 1
0 1 1 0
1 0 1 0
1 0 0 1
```

So the detector tries all four rotations:

```python
candidates = [
    bits,
    rotate90(bits),
    rotate180(bits),
    rotate270(bits),
]

for rotated_bits in candidates:
    compare_to_dictionary(rotated_bits)
```

The best matching rotation gives:

```text
marker ID
marker orientation
correct corner ordering
```

That corner ordering is important for pose. The clever thing here is because after the contour is found, the marker is already perspective-warped into a square. So it's sufficient to try 4 rotates to find the best matching contour

## Identify Marker ID Using Dictionary matching with Hamming distance

The detector compares the decoded bit matrix to known dictionary entries.  Suppose the dictionary has this marker ID 17:

```text
dictionary marker 17

1 0 0 1
0 1 1 0
1 1 0 0
0 0 1 1
```

Now suppose one bit was misread:

```text
observed with one bad bit

1 0 0 1
0 1 1 0
1 0 0 0   <- one bit changed
0 0 1 1
```

Compare to dictionary marker 17:

```text
different bits = 1
```

If the dictionary allows correction up to that error threshold, it can still identify the marker.

That is why dictionary design matters. If two valid markers are too similar, a noisy observation could be confused. OpenCV’s docs describe the inter-marker distance as the minimum Hamming distance between dictionary markers, and that distance determines error-detection and error-correction capability. ([OpenCV Documentation][1])

---

# Step 5 - Pose estimation from four corners

After decoding, we know the 2D image corners:

```text
image corners, pixels

u0, v0
u1, v1
u2, v2
u3, v3
```

And because the printed marker has known physical size, we know the 3D marker-frame corners. For a marker of side length `L`, define marker coordinates:

```python
P0 = [-L/2,  L/2, 0]
P1 = [ L/2,  L/2, 0]
P2 = [ L/2, -L/2, 0]
P3 = [-L/2, -L/2, 0]
```

The marker is planar, so all points have:

```text
Z = 0
```

The camera projection model is:

```text
s [u, v, 1]^T = K [R | t] [X, Y, Z, 1]^T
```

where:

```text
K = camera intrinsics
R = marker rotation relative to camera
t = marker translation relative to camera
```

Since the marker lies on a plane, `Z = 0`, so:

```text
s [u, v, 1]^T = K [r1 r2 t] [X, Y, 1]^T
```

This is a homography:

```text
s [u, v, 1]^T = H [X, Y, 1]^T
```

where:

```text
H = K [r1 r2 t]
```

So a square marker gives four 2D-3D correspondences, enough to estimate pose. OpenCV’s ArUco docs emphasize that one benefit of binary square fiducial markers is that a single marker provides enough corner correspondences to estimate camera pose. ([GitHub][2])

In practice, OpenCV usually calls a PnP solver:

```python
ok, rvec, tvec = cv2.solvePnP(
    object_points,   # 3D marker corners
    image_points,    # detected 2D corners
    K,
    dist_coeffs,
)
```

Then:

```python
R, _ = cv2.Rodrigues(rvec)
```

The result is usually interpreted as:

```text
camera_T_marker
```

or:

```text
marker pose in camera frame
```

Meaning:

```text
P_camera = R * P_marker + t
```

---

# 9. Small numerical pose example

Say the marker side length is:

```python
L = 0.10  # meters
```

So marker-frame corners are:

```python
P0 = [-0.05,  0.05, 0]
P1 = [ 0.05,  0.05, 0]
P2 = [ 0.05, -0.05, 0]
P3 = [-0.05, -0.05, 0]
```

Suppose camera intrinsics are:

```python
fx = 600
fy = 600
cx = 320
cy = 240
```

So:

```python
K = [
    [600,   0, 320],
    [  0, 600, 240],
    [  0,   0,   1],
]
```

If the marker is front-facing at:

```python
t = [0, 0, 0.5]  # 50 cm in front of camera
R = identity
```

Then projection is:

```text
u = fx * X/Z + cx
v = fy * Y/Z + cy
```

For corner `P0 = [-0.05, 0.05, 0]`, camera point is:

```python
Xc = -0.05
Yc =  0.05
Zc =  0.50
```

So:

```python
u = 600 * (-0.05 / 0.50) + 320 = 260
v = 600 * ( 0.05 / 0.50) + 240 = 300
```

Do the same for all corners:

```text
P0 -> (260, 300)
P1 -> (380, 300)
P2 -> (380, 180)
P3 -> (260, 180)
```

So a 10 cm marker at 50 cm distance appears as:

```text
width in image = 120 pixels
```

because:

```text
pixel_width = fx * physical_width / depth
            = 600 * 0.10 / 0.50
            = 120 px
```

PnP solves the inverse problem:

```text
Given:
    3D marker corners
    detected 2D image corners
    camera intrinsics

Find:
    R, t
```

---

# 10. Compact pseudocode

```python
def detect_aruco_and_pose(image, K, dist_coeffs, marker_length, dictionary):
    gray = to_grayscale(image)

    # 1. Find square candidates
    binary = adaptive_threshold(gray)
    contours = find_contours(binary)

    candidates = []
    for contour in contours:
        poly = approximate_polygon(contour)

        if len(poly) == 4 and is_convex(poly) and area(poly) > min_area:
            candidates.append(order_corners(poly))

    detections = []

    # 2. Decode each candidate
    for corners in candidates:
        canonical = perspective_warp(gray, corners, output_size=(N, N))

        bits_with_border = sample_cells(canonical)

        if not black_border_is_valid(bits_with_border):
            continue

        inner_bits = remove_border(bits_with_border)

        best_id = None
        best_rotation = None
        best_distance = infinity

        for rotation in [0, 90, 180, 270]:
            rotated = rotate_bits(inner_bits, rotation)

            for marker_id, dict_bits in dictionary:
                d = hamming_distance(rotated, dict_bits)

                if d < best_distance:
                    best_distance = d
                    best_id = marker_id
                    best_rotation = rotation

        if best_distance <= allowed_error:
            corrected_corners = rotate_corner_order(corners, best_rotation)
            detections.append((best_id, corrected_corners))

    # 3. Estimate pose for each decoded marker
    poses = []
    object_points = marker_3d_corners(marker_length)

    for marker_id, image_corners in detections:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_corners,
            K,
            dist_coeffs,
        )

        if ok:
            poses.append((marker_id, rvec, tvec))

    return poses
```

[1]: https://docs.opencv.org/4.13.0/d5/dae/tutorial_aruco_detection.html?form=MG0AV3&utm_source=chatgpt.com "Detection of ArUco Markers - OpenCV"
[2]: https://github.com/jing-vision/opencv-aruco/blob/master/tutorials/aruco_detection/aruco_detection.markdown?utm_source=chatgpt.com "opencv-aruco/tutorials/aruco_detection/aruco_detection ... - GitHub"
