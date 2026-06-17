---
layout: post
title: Computer Vision - Charuco Board Detection
date: '2026-03-16 13:19'
subtitle: 
comments: true
header-img: img/post-bg-infinity.jpg
tags:
    - Deep Learning
---

## ChArUco Pose Estimation

A ChArUco board is basically a chessboard combined with ArUco markers.

An **ArUco marker** is a square fiducial marker with a binary ID pattern inside it:

```text
marker id=51
+---------+
| black   |
| pattern |
| id bits |
+---------+
```

An **ArUco board** lets OpenCV detect square marker IDs and corners directly. A **ChArUco board** goes one step further: OpenCV detects the ArUco markers first, then uses the known board layout to infer the chessboard intersection corners between the markers. Its pipeline is roughly:

1. detect ArUco markers
2. use marker IDs to understand board layout
3. interpolate/find chessboard corners
4. use those corners for calibration or pose estimation

This matters because marker corners are useful, but they are not always the most accurate features. The marker border is thick, and the internal binary pattern is designed for robust ID decoding, not necessarily for the best subpixel corner localization.

ChArUco corners are chessboard intersections, so they are usually better 2D features for pose estimation.

The goal of the whole pipeline is to build accurate 2D-to-3D correspondences:

```text
known 3D board point  <-->  detected 2D image point
```

Then `solvePnP` estimates the pose that best explains those correspondences.

## Step 1 - Detect ArUco Corners

When OpenCV detects ArUco markers, it returns marker IDs and four image corners per marker:

```python
marker_corners, marker_ids, rejected = detector.detectMarkers(gray)

# marker_ids:
#   [51, 56, 58, ...]

# marker_corners:
#   4 image points per marker
```

Each detected marker gives four 2D pixel corners:

```text
id=51 corner 0
id=51 corner 1
id=51 corner 2
id=51 corner 3
```

For ChArUco, the detector gives both the raw ArUco detections and the interpolated ChArUco corners:

```python
charuco_corners, charuco_ids, marker_corners, marker_ids = (
    charuco_detector.detectBoard(gray)
)

# marker_corners, marker_ids:
#   raw ArUco marker detections

# charuco_corners, charuco_ids:
#   detected/interpolated ChArUco chessboard corners
```

The runtime flow is:

```text
camera image
  ↓
detect ArUco markers
  ↓
use marker IDs to determine board position/orientation
  ↓
interpolate visible ChArUco chessboard corners
  ↓
match ChArUco corner IDs to known 3D board points
  ↓
solve board pose with solvePnP / solvePnPRansac
```

So ChArUco depends on ArUco. You usually do not manually choose “ArUco or ChArUco” at runtime. If you call `CharucoDetector.detectBoard()`, the ArUco detection step is part of the ChArUco detection process.

## Step 2 - Matching Image Points to Board Points

For pose estimation, the important outputs are:

```text
charuco_ids
charuco_corners
```

Each ChArUco corner ID corresponds to a known 3D point on the physical board.

At runtime, this call:

```python
object_points, image_points = self.board.matchImagePoints(
    charuco_corners,
    charuco_ids,
)
```

means:

```text
For each detected ChArUco corner ID:
    find the known 3D location of that corner on the board
    pair it with the detected 2D pixel location in the image
```

The result is a matched list:

```text
object_points[i]  <-->  image_points[i]
```

For example:

```text
object_points[0] = [0.045, 0.045, 0.000] meters
image_points[0]  = [612.3, 381.7] pixels

object_points[1] = [0.090, 0.045, 0.000] meters
image_points[1]  = [655.8, 379.9] pixels
```

The object points are 3D coordinates in the board frame. Since the board is planar, their `z` values are usually zero:

```text
X_board = [x, y, 0]
```

The image points are 2D pixel coordinates:

```text
u, v = pixel location in the camera image
```

## Step 4 - What `solvePnP` Solves

Given:

```text
3D board points
matching 2D image points
camera intrinsics K
distortion coefficients D
```

`solvePnP` estimates:

```text
rvec, tvec
```

These describe the rigid transform from the board frame to the camera frame.

In equation form:

```text
X_camera = R * X_board + t
```

Then OpenCV projects the camera-frame 3D point into the image:

```text
X_board -> X_camera -> pixel coordinate
```

The solver chooses `R` and `t` so that the projected pixels land close to the detected ChArUco pixels:

```text
projected pixel ≈ detected ChArUco pixel
```

This is the reprojection-error idea.

### Pose Direction: The Important Part

The subtle but important part is the direction of the returned pose.

OpenCV returns:

```text
rvec, tvec = transform from object/board frame to camera frame
```

Meaning:

```python
X_camera = R @ X_board + t
```

Which is not directly:

```text
the camera origin expressed in the board frame
```

### What RANSAC Adds

Plain `solvePnP()` uses all point correspondences. If one or two ChArUco corners are wrong, the final pose can be pulled away from the correct solution.

`solvePnPRansac()` is more robust because it tries to find a pose that agrees with most of the points while rejecting outliers.

A simplified mental model is:

```python
best_pose = None
best_inliers = []

for trial in range(100):
    # 1. sample a small subset of 2D/3D matches
    # 2. solve a candidate pose from that subset
    # 3. project all 3D points into the image
    # 4. measure reprojection error for every point
    # 5. keep points whose error is small enough

    if len(inliers) > len(best_inliers):
        best_pose = candidate_pose
        best_inliers = inliers

return best_pose, best_inliers
```

The `reprojectionError` threshold controls how strict the inlier test is.

For example, if:

```python
max_reproj_error_px = 2.0
```

then a point is considered an inlier if:

```text
distance(projected_pixel, detected_pixel) < 2 px
```

Mental model:

```text
solvePnP:
    fit pose using all points

solvePnPRansac:
    find a pose while rejecting bad point matches
```

## What `solvePnPRefineLM()` Does

After RANSAC finds a good inlier set, you can refine the pose:

```python
cv2.solvePnPRefineLM(
    object_points[inlier_indices],
    image_points[inlier_indices],
    camera_matrix,
    dist_coeffs,
    rvec,
    tvec,
)
```

This runs a local nonlinear optimization. It slightly adjusts `rvec` and `tvec` to reduce reprojection error on the inlier points.

## Important Warning About Custom Board Frames

`matchImagePoints()` returns object points in the coordinate frame used by the OpenCV board object.

So if `self.board` is a normal OpenCV `CharucoBoard`, the returned `object_points` are in OpenCV’s board frame. They are not automatically converted into your custom bottom-left checker frame.

This means:

```python
object_points, image_points = self.board.matchImagePoints(
    charuco_corners,
    charuco_ids,
)
```

does not automatically use your custom `board_point_to_checker()` convention. You can use `matchImagePoints()` as-is:

```python
object_points, image_points = self.board.matchImagePoints(
    charuco_corners,
    charuco_ids,
)
```

Then the returned pose means:

```text
OpenCV board frame -> camera frame
```

or:

```text
T_camera_board_opencv
```

This is easiest if you only need a consistent board pose and do not care where the board origin is placed physically.

## OpenCV Version Warning

If you generate and print a ChArUco board with one OpenCV version, then detect it with another, make sure the board pattern convention is compatible.

This is especially worth checking for boards generated before OpenCV 4.6.0. If the physical printed board was generated with the older convention, you may need to enable the legacy pattern setting when constructing the board.

The practical rule is:

```text
The printed board layout and the OpenCV board object must agree.
```

If they do not agree, marker detection may still work, but the ChArUco corner IDs or board geometry can be inconsistent, which can break pose estimation.
