---
layout: post
title: Computer Vision - RANSAC
date: 2021-02-01 13:19
subtitle:
comments: true
tags:
  - Computer Vision
---

## Introduction

**RANSAC** means: RANdom SAmple Consensus. It is a robust fitting method for data with outliers. Normal least-squares fitting asks: `What model minimizes total error over all points?` RANSAC asks: `Can I find a small subset of points that creates a model
that many other points agree with?` That is useful when some measurements are wrong.

For example, in ChArUco/PnP: Most detected corners are correct. Some detected corners may be wrong, mismatched, blurry, or noisy. RANSAC tries many small random subsets, estimates a model from each subset, and keeps the model with the most **inliers**.

## Motivating Example

Suppose you want to fit a line through points. Good points roughly lie on `y = 2x + 1`
bad points are random outliers

```text
x   y
0   1.1
1   3.0
2   5.2
3   7.1
4   9.0
5   20.0   <- outlier
6  -3.0    <- outlier
```

Least squares may get pulled by the outliers. RANSAC does this instead:

```text
repeat many times:
    randomly choose minimal number of points
    fit model from those points
    count how many all points agree with that model
keep the model with the largest agreement
refit model using only agreed/inlier points
```

For a 2D line, the minimal sample size is 2 points. Set an inlier threshold:

```python
threshold = 0.5
```

A point is an inlier if its distance from the line is less than `0.5`.

### Trial 1: choose two good points

Randomly choose:

```text
(1, 3.0) and (4, 9.0)
```

Fit line:

```text
slope = (9.0 - 3.0) / (4 - 1)
      = 6 / 3
      = 2

intercept = 3.0 - 2 * 1
          = 1
```

So candidate model:

```text
y = 2x + 1
```

Now score every point. For each point:

```text
predicted_y = 2x + 1
error = |observed_y - predicted_y|
```

```text
point       predicted     error      inlier?
(0, 1.1)    1.0          0.1        yes
(1, 3.0)    3.0          0.0        yes
(2, 5.2)    5.0          0.2        yes
(3, 7.1)    7.0          0.1        yes
(4, 9.0)    9.0          0.0        yes
(5,20.0)   11.0          9.0        no
(6,-3.0)   13.0         16.0        no
```

5 inliers, Good model.

### Trial 2: choose one good point and one outlier

Randomly choose:

```text
(1, 3.0) and (5, 20.0)
```

Fit line:

```text
slope = (20.0 - 3.0) / (5 - 1)
      = 17 / 4
      = 4.25

intercept = 3.0 - 4.25 * 1
          = -1.25
```

Candidate model:

```text
y = 4.25x - 1.25
```

Score points:

```text
point       predicted       error
(0, 1.1)    -1.25           2.35    no
(1, 3.0)     3.00           0.00    yes
(2, 5.2)     7.25           2.05    no
(3, 7.1)    11.50           4.40    no
(4, 9.0)    15.75           6.75    no
(5,20.0)    20.00           0.00    yes
(6,-3.0)    24.25          27.25    no
```

2 inliers, Bad model. RANSAC keeps the model from Trial 1 because more points agree with it.

# How many random trials do I need?

Let:

- w = probability that one random data point is an inlier
- s = sample size needed to fit one model
- N = number of RANSAC iterations
- p = desired probability of seeing at least one all-inlier sample

For one random sample of size `s`, `probability all sampled points are inliers = w^s`. Therefore: `probability sample is bad = 1 - w^s`. After `N` independent trials: `probability all samples are bad = (1 - w^s)^N` So probability at least one sample is good: `p = 1 - (1 - w^s)^N`. Solve for `N`:

```text
N = log(1 - p) / log(1 - w^s)
```

That is the classic RANSAC iteration formula.

## Example

Suppose: 70% of points are inliers, w = 0.7, line fitting needs 2 points, so s = 2. Desired success probability: p = 0.99.

Then:

```text
w^s = 0.7^2 = 0.49
```

Probability one random pair is all-inlier: 49%

Iterations needed:

```text
N = log(1 - 0.99) / log(1 - 0.49)
  = log(0.01) / log(0.51)
  ≈ -4.605 / -0.673
  ≈ 6.84
```

So about: 7 iterations

## Pseudocode

```python
def ransac(data, fit_model, compute_error, sample_size, threshold, max_iters):
    best_model = None
    best_inliers = []

    for _ in range(max_iters):
        # 1. Randomly sample minimal subset
        sample = random_sample(data, sample_size)

        # 2. Fit candidate model
        model = fit_model(sample)

        if model is None:
            continue

        # 3. Score model against all data
        inliers = []

        for point in data:
            error = compute_error(model, point)

            if error < threshold:
                inliers.append(point)

        # 4. Keep best consensus set
        if len(inliers) > len(best_inliers):
            best_model = model
            best_inliers = inliers

    # 5. Refit using all inliers
    if best_model is not None and len(best_inliers) >= sample_size:
        best_model = fit_model(best_inliers)

    return best_model, best_inliers
```

For pnp pose:

```text
repeat:
    randomly select a small subset of 3D-2D correspondences
    estimate candidate pose R,t
    project all 3D points using R,t
    compute reprojection error for every point
    count inliers
keep pose with most inliers
refine pose using all inliers
```

Projection math:

```text
P_camera = R P_object + t
```

Then:

```text
u_pred = fx * X_camera / Z_camera + cx
v_pred = fy * Y_camera / Z_camera + cy
```

Reprojection error:

```text
e_i = sqrt((u_i - u_pred_i)^2 + (v_i - v_pred_i)^2)
```

Inlier rule:

```text
e_i < reprojectionError
```

So if `reprojectionError=3.0`, then a detected corner is an inlier if the predicted projected corner lands within 3 pixels of the measured corner.
