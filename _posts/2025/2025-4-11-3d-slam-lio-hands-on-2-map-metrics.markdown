---
layout: post
title: Robotics - [3D SLAM - 5] Map Metrics
date: 2025-4-8 13:19
subtitle: Gaussian Entropy, Graph Tension, Chi Squared, Trajectory Smoothness
header-img: img/post-bg-o.jpg
tags:
  - Robotics
  - SLAM
comments: true
---

## Residual Calculation

In scan matching, a residual is the raw error term `e`. The weighted squared error is:

```cpp
chi2 = e.transpose() * information * e;
```

where:

```cpp
information = covariance.inverse()
```

So if the uncertainty/covariance is larger, the information weight is smaller, and the same residual contributes less to the final cost.

`chi_square_per_dof` is the normalized average:

```cpp
chi_square_per_dof = chi2 / dof;
```

where `dof` is the residual dimension.

---

### NDT Residuals

In NDT, the target cloud is represented as voxels. Each valid voxel stores a Gaussian distribution:

```cpp
voxel_mean
voxel_covariance
voxel_information = voxel_covariance.inverse()
```

An effective point is a transformed source point that lands in a valid target voxel.

For source point `p_i` transformed by pose `T`:

```cpp
p = T * p_i
```

the residual is:

```cpp
e = p - voxel_mean
```

The NDT chi-square cost is:

```cpp
chi2 = e.transpose() * voxel_information * e;
```

This is also the squared Mahalanobis distance from the point to the voxel Gaussian.

---

### ICP Residuals

In point-to-X ICP, a valid point must have a valid correspondence in the target cloud. For line or plane ICP, the target neighborhood must contain enough nearby points to estimate a stable geometric feature.

Common ICP residuals:

#### Point-to-point

```cpp
e = p - q
```

where `p` is the transformed source point and `q` is the matched target point.

#### Point-to-plane

```cpp
e = n.transpose() * (p - q)
```

where `n` is the target plane normal.

This is the signed distance from the point to the plane.

#### Point-to-line

```cpp
e = distance(p, line)
```

For a line through points `a` and `b`:

```cpp
e = ||(p - a).cross(p - b)|| / ||a - b||
```

The chi-square cost is still:

```cpp
chi2 = e.transpose() * information * e;
```

or, for a scalar residual:

```cpp
chi2 = e * information * e;
```

---

## Gaussian Entropy

Gaussian entropy is the actual information-theory differential entropy of a Gaussian distribution.

For a Gaussian:

```cpp
x ~ N(mu, Sigma)
```

the differential entropy is:

```cpp
H = 0.5 * log((2 * pi * exp(1))^d * det(Sigma))
```

where:

```cpp
d      = dimension
Sigma  = covariance matrix
det    = determinant
```

For a 3D point distribution, `Sigma` is a `3x3` covariance matrix:

```cpp
Sigma =
[ variance_x      covariance_xy  covariance_xz
  covariance_yx   variance_y     covariance_yz
  covariance_zx   covariance_zy  variance_z    ]
```

Differential entropy means:

```cpp
H = -∫ p(x) log(p(x)) dx
```

For 3D:

```cpp
H = -∫∫∫ p(x, y, z) log(p(x, y, z)) dx dy dz
```

So in NDT, we treat the points inside a voxel as samples from a continuous 3D Gaussian distribution.

### Prep 1 - Covariance Cleanup

Before computing the determinant, clean up the covariance matrix:

```cpp
cov = 0.5 * (cov + cov.transpose()) + epsilon * I;
```

This does two things:

1. Makes the covariance symmetric again after floating-point operations.

2. Adds a small diagonal regularization term so the matrix is less likely to be singular.

This is useful because entropy requires:

```cpp
log(det(cov))
```

If the determinant is zero or negative, the entropy becomes invalid.

Do not use:

```cpp
cov = 0.5 * (cov.transpose() * cov) + epsilon * I;
```

because that changes the meaning of the covariance.

### Prep 2 - Determinant Interpretation

For covariance:

```cpp
det(cov) = lambda1 * lambda2 * lambda3
```

where `lambda1`, `lambda2`, and `lambda3` are the eigenvalues of the covariance matrix.

The eigenvectors define the three principal axes of the point distribution. The eigenvalues describe the spread along those axes.

In an NDT voxel:

- Points evenly distributed in 3D produce three significant eigenvalues.

- Points lying mostly on a plane produce two large eigenvalues and one small eigenvalue.

- Points lying mostly on a line produce one large eigenvalue and two small eigenvalues.

So `det(cov)` measures the volume of the covariance ellipsoid.

Low determinant means the voxel is geometrically degenerate in at least one direction.

For SLAM map metrics, entropy can be aggregated across voxels:

```cpp
mean_entropy = sum(voxel_entropy) / valid_voxel_count
```

or weighted by point count:

```cpp
weighted_mean_entropy = sum(point_count_i * entropy_i) / sum(point_count_i)
```

Low entropy usually means a sharper or more compact voxel distribution, but it can also indicate geometric degeneracy. So entropy should be used together with coverage, residual RMSE, chi-square, and eigenvalue-ratio metrics.

---

## Edge Tension

Edge tension is the weighted squared error of one pose-graph edge.

```cpp
edge_tension = error.transpose() * information * error;
```

This is the chi-square value of the edge.

It is also the squared Mahalanobis distance if:

```cpp
information = covariance.inverse()
```

So:

```cpp
edge_tension = chi2 = mahalanobis_distance^2
```

Graph tension aggregates edge tensions across the pose graph. For example:

```cpp
mean_graph_tension = mean(edge_tension_i)
rms_graph_tension  = sqrt(mean(edge_tension_i^2))
max_graph_tension  = max(edge_tension_i)
```

High graph tension means some edges disagree strongly with the current pose estimate.

---

### Trajectory Smoothness Stats

Trajectory smoothness measures how much the estimated trajectory changes abruptly over time.

Given timestamped poses, compute:

1. Translational change.

2. Angular change.

3. Linear velocity.

4. Angular velocity.

5. Linear acceleration.

6. Angular acceleration.

A smooth trajectory should have reasonable acceleration values. Large spikes may indicate pose jumps, bad scan matching, bad loop closures, or timestamp issues.

A practical score can combine linear and angular acceleration:

```cpp
score = mean_linear_acceleration_sq
      + angular_weight * mean_angular_acceleration_sq;
```

Higher score means a less smooth trajectory.
