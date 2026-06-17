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

## Charuco

ChArUco is basically **chessboard + ArUco markers**. Aruco board:  Detect square fiducial markers directly. ChArUco board: Detect ArUco markers first, then use them to infer/interpolate chessboard corners. An **ArUco marker** is one black square marker with a binary ID pattern inside.

```text
marker id=51
+---------+
| black   |
| pattern |
| id bits |
+---------+
```

When OpenCV detects ArUco, it gives you:

```python
marker_corners, marker_ids, rejected = detector.detectMarkers(gray)
# marker_ids: [51, 56, 58, ...]

# marker_corners: 4 image points per marker
```

Each marker gives four corners. These corners are useful, but they are relatively coarse because the marker border is thick and the inner binary pattern is not ideal for super-accurate subpixel localization.

A **ChArUco board** adds chessboard intersections between the ArUco markers. black/white checkerboard squares; some squares contain ArUco IDs; checker intersections become ChArUco corners.  OpenCV first detects ArUco markers, then uses the known board layout to figure out which chessboard corners are visible. So this:

```python
charuco_corners, charuco_ids, marker_corners, marker_ids = (
    charuco_detector.detectBoard(gray)
)
# marker_corners, marker_ids: raw ArUco marker detections
# charuco_corners, charuco_ids: inferred chessboard corner detections
```


```text
camera image
  ↓
detect ArUco markers
  ↓
use marker IDs to understand board location/orientation
  ↓
interpolate/find chessboard corners
  ↓
use ChArUco corners for pose estimation
```

So **ChArUco depends on ArUco**. You usually do **not** choose one or the other manually. `CharucoDetector.detectBoard()` internally does the ArUco step for you and then gives you both outputs.

For pose estimation, you want accurate 2D-to-3D correspondences:

```text
3D board point  <->  2D image point
```

ArUco gives you marker corners:

```text
id=51 corner 0
id=51 corner 1
id=51 corner 2
id=51 corner 3
```

ChArUco gives you chessboard intersections:

```text
charuco corner id=0
charuco corner id=1
charuco corner id=2
...
```

Chessboard corners are usually more precise than marker corners. So for pose, this is preferred:

```python
object_points = board_corners[charuco_id_indices]
image_points = charuco_corners.reshape(-1, 2)

solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
```

## During Runtime
```python
object_points, image_points = self.board.matchImagePoints(
    charuco_corners,
    charuco_ids,
)
```

means: For each detected ChArUco corner ID:  find its known 3D location on the physical board
pair it with its detected 2D pixel location in the image. So the result is a matched list:

```text
object_points[i]  <-->  image_points[i]

# for example"
object_points[0] = [0.045, 0.045, 0.000] meters
image_points[0]  = [612.3, 381.7] pixels

object_points[1] = [0.090, 0.045, 0.000] meters
image_points[1]  = [655.8, 379.9] pixels
```


Then, Given: 3D board points in object/checker frame, matching 2D pixel points in camera image, camera intrinsics K,  distortion coefficients D. Find: rotation rvec, translation tvec. Such that:
  projected 3D points land close to the observed 2D pixels: 

```text
X_camera = R * X_object + t
```

Then OpenCV projects `X_camera` through the camera matrix:

```text
X_object -> X_camera -> pixel coordinate
```

The solver chooses `R` and `t` so that:

```text
projected pixel ≈ detected ChArUco pixel
```


Direction of the returned pose is the subtle but important part. OpenCV returns:

```text
rvec, tvec = transform from object/board frame to camera frame
```

Meaning:

```python
X_camera = R @ X_board + t
```

So `tvec` is:

```text
the board/checker origin expressed in the camera frame
```

It is **not** directly:

```text
camera origin expressed in the board frame
```


##  What does RANSAC add?

Plain `solvePnP()` uses all points. If one or two ChArUco corners are wrong, the pose can be pulled badly.

`solvePnPRansac()` does something like:

```python
best_pose = None
best_inliers = []

for trial in range(100):
    sample a small random subset of point matches
    solve a candidate pose from that subset

    project all 3D object_points into the image
    measure pixel error to image_points

    inliers = points with error < reprojectionError

    if len(inliers) > len(best_inliers):
        best_pose = candidate_pose
        best_inliers = inliers

return best_pose, best_inliers
```

So `reprojectionError` means a point is considered an inlier if its projected pixel is within this many pixels of the detected image point., For example, if `max_reproj_error_px = 2.0`, then a matched corner is an inlier if:

```text
distance(projected_pixel, detected_pixel) < 2 px
```

## What `solvePnPRefineLM()` does

After RANSAC finds a good inlier set, this part:

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

does a local nonlinear optimization. It slightly adjusts `rvec` and `tvec` to reduce reprojection error on the inlier points.

Mental model:

```text
RANSAC:
  robustly find a good pose despite bad points

LM refinement:
  polish that pose using the good points
```

## 8. Important warning about your bottom-left frame

`matchImagePoints()` returns object points in the coordinate frame used internally by `self.board`.

So if your `self.board` is a normal OpenCV `CharucoBoard`, the returned `object_points` are in the OpenCV board frame, not necessarily your custom bottom-left checker frame.

This means:

```python
object_points, image_points = self.board.matchImagePoints(...)
```

does **not** automatically use your custom `board_point_to_checker()` convention.

So you have two choices.

### Option A: use OpenCV board frame internally

Use `matchImagePoints()` as-is. Then the returned pose is:

```text
OpenCV board frame -> camera frame
```

This is easiest.

### Option B: convert object points to your checker frame before `solvePnP`

If you want the returned `rvec, tvec` to mean:

```text
bottom-left checker frame -> camera frame
```

then convert the object points before calling `solvePnPRansac()`:

```python
object_points, image_points = self.board.matchImagePoints(
    charuco_corners,
    charuco_ids,
)

object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)

object_points_checker = np.asarray(
    [self.board_point_to_checker(p) for p in object_points],
    dtype=np.float32,
)

ok, rvec, tvec, inliers = cv2.solvePnPRansac(
    object_points_checker,
    image_points,
    camera_matrix,
    dist_coeffs,
    iterationsCount=100,
    reprojectionError=self.config.max_reproj_error_px,
    confidence=0.99,
    flags=cv2.SOLVEPNP_ITERATIVE,
)
```

Then `rvec, tvec` now describe:

```text
checker frame -> camera frame
```

where `checker frame` is your bottom-left-origin frame.

## Bottom line

Your guess should be slightly corrected:

```text
Given camera intrinsics, 2D image points, and their known 3D board coordinates,
solvePnPRansac solves for board/checker -> camera transform.
```

So:

```text
rvec/tvec tell you where the board origin is in the camera frame.
```

If you want:

```text
where the camera is in the board/checker frame
```

then invert `R, t`.