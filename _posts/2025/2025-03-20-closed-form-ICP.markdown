---
layout: post
title: Robotics - [3D SLAM - 7] Closed Form ICP
date: 2025-03-17 13:19:00
subtitle:
header-img: img/post-bg-o.jpg
tags:
  - Robotics
  - SLAM
comments: true
---

---

## Rigid Alignment Form ICP

 **closed-form rigid alignment step** inside ICP. The outer loop is ICP because it repeatedly updates correspondences, but the per-iteration pose update is solved by **SVD/Kabsch/Umeyama-style absolute orientation**, not by linearizing residuals with Jacobians.

## ICP refinement: CAD/model points + observed object cluster

For each coarse pose candidate ${}^{checker}T_{object}$, we refine it by aligning target point cloud to the observed segmented object cluster.

Let:

```text
P = CAD/model points in object frame
Q = observed object points in checker frame
T = current estimate of T_checker_object
```

At each ICP iteration, we first transform the CAD points into the checker frame:

$$  
p_i^{checker} = T p_i^{object}  
$$

Then for each observed point $q_j \in Q$, find its nearest transformed CAD point:

$$  
i^*(j)

\arg\min_i  
\left|  
q_j - p_i^{checker}  
\right|_2  
$$

This gives correspondence pairs:

$$  
\left(  
p_{i^*(j)}^{checker},  
q_j  
\right)  
$$

The code then rejects correspondences whose distance is too large:

$$  
\left|  
q_j - p_{i^*(j)}^{checker}  
\right|_2  
\le d_{max}  
$$

and optionally keeps only the best trimmed fraction of the remaining correspondences.

---

## Closed-form rigid transform update

Now suppose we have $N$ valid correspondences:

$$  
(a_k, b_k), \quad k = 1,\dots,N  
$$

where:

```text
a_k = transformed CAD/source point
b_k = observed target point
```

We want to find a rigid transform $(R, t)$ that best maps the source points to the target points:

$$  
\min_{R,t}  
\sum_{k=1}^{N}  
\left|  
R a_k + t - b_k  
\right|^2  
$$

subject to:

$$  
R^T R = I,  
\qquad  
\det(R)=1  
$$

This is the classic point-to-point rigid alignment problem.

---

## Step 1: Compute centroids

 $$  
\bar{a} =

\frac{1}{N}  
\sum_{k=1}^{N}  
a_k  
$$

 $$  
\bar{b} =

\frac{1}{N}  
\sum_{k=1}^{N}  
b_k  
$$

---

## Step 2: Center the points

$$  
\tilde{a}_k = a_k - \bar{a}  
$$

$$  
\tilde{b}_k = b_k - \bar{b}  
$$

In your code this is:

```python
source_centroid = np.mean(source_points, axis=0)
target_centroid = np.mean(target_points, axis=0)

source_centered = source_points - source_centroid
target_centered = target_points - target_centroid
```

---

## Step 3: Build the cross-covariance matrix

Your code does:

```python
covariance = source_centered.T @ target_centered
```

Mathematically:

$$  
H =

\sum_{k=1}^{N}  
\tilde{a}_k  
\tilde{b}_k^T  
$$

or in matrix form:

$$  
H =

A^T B  
$$

where $A$ is the matrix of centered source points and $B$ is the matrix of centered target points.

Important detail: this is called a covariance-like matrix, but the scale factor $\frac{1}{N}$ is not necessary for the rotation, because it does not change the SVD singular vectors.

---

## Step 4: SVD of the cross-covariance

Your code does:

```python
u, _, vt = np.linalg.svd(covariance)
rotation = vt.T @ u.T
```

Mathematically:

$$  
H = U \Sigma V^T  
$$

Then:

$$  
R = V U^T  
$$

In NumPy, `vt` is $V^T$, so:

```python
rotation = vt.T @ u.T
```

means:

$$  
R = V U^T  
$$

This is the optimal least-squares rotation for the current correspondences.

---

## Step 5: Reflection correction

**Sometimes SVD can produce a reflection instead of a valid rotation**:

$$  
\det(R) < 0  
$$

A true 3D rotation should satisfy:

$$  
\det(R) = +1  
$$

If:

$$  
\det(R) < 0  
$$

then the code flips the last row of $V^T$:

```python
if np.linalg.det(rotation) < 0.0:
    vt[-1, :] *= -1.0
    rotation = vt.T @ u.T
```

Mathematically, this is equivalent to inserting a correction matrix:

$$  
D =  
\operatorname{diag}(1, 1, \det(VU^T))  
$$

and using:

$$  
R = V D U^T  
$$

This prevents the fitted transform from mirroring the CAD model.

---

## Step 6: Solve translation

Once $R$ is known, translation is:

$$  
t = \bar{b} - R \bar{a}  
$$

In code:

```python
transform[:3, 3] = target_centroid - rotation @ source_centroid
```

So the incremental transform is:

$$  
\Delta T =  
\begin{bmatrix}  
R & t \  
0 & 1  
\end{bmatrix}  
$$

Then ICP updates the current object pose by left-multiplying:

$$  
T \leftarrow \Delta T \ T  
$$

In your code:

```python
transform = delta @ transform
```

That means:

```text
delta maps the currently transformed CAD points closer to the observed points.
```

---

## Why this is still ICP

ICP has two alternating steps:

```text
1. Correspondence step:
   find nearest neighbors between current source and target

2. Alignment step:
   solve best rigid transform for those correspondences
```

Your code does exactly that.

But there are two common ways to do the alignment step:

```text
A. Closed-form point-to-point ICP:
   use SVD to solve R,t exactly for fixed correspondences

B. Jacobian/Gauss-Newton ICP:
   linearize residuals and solve normal equations
```

Your code is **A**, not **B**.

---

## Does this assume small rotation?

No — the SVD rigid alignment step itself does **not** require small rotation.

For fixed correspondences, the SVD solution finds the globally optimal rigid rotation, even if the rotation is large.

The part that _does_ need a good initial pose is the **nearest-neighbor correspondence step**. If the initial pose is far off, nearest-neighbor matching can pair the wrong CAD points with the wrong observed points. Then SVD will solve the wrong alignment problem.

So the right statement is:

```text
The SVD update does not assume small rotation for fixed correspondences.
ICP as a whole still needs a decent initial pose because correspondences depend on the current pose.
```

That is why your code generates many yaw candidates first. The coarse yaw search tries to put the CAD model close enough that nearest-neighbor correspondences are meaningful.

---

## Difference from Jacobian point-to-point ICP

A Jacobian-based point-to-point ICP would define residuals:

# $$  

r_k(\xi)

R(\xi) a_k + t(\xi) - b_k  
$$

where $\xi \in \mathfrak{se}(3)$ is a small pose update.

Then it linearizes:

$$  
r_k(\xi + \delta \xi)  
\approx  
r_k(\xi) + J_k \delta \xi  
$$

and solves:

# $$  
\delta \xi

\arg\min_{\delta \xi}  
\sum_k  
\left|  
r_k + J_k \delta \xi  
\right|^2  
$$

which leads to normal equations:

# $$  

\left(  
\sum_k J_k^T J_k  
\right)  
\delta \xi

\sum_k J_k^T r_k  
$$

Then:

$$  
T \leftarrow \exp(\delta \xi^\wedge) T  
$$

Your code does **not** do this. It does not compute $J$, $J^T J$, or a Lie algebra update. It directly solves $(R,t)$ from point correspondences using SVD.

---

## Concise version for your notes

```text
ICP refinement is implemented as closed-form point-to-point ICP.

For each coarse T_checker_object candidate:
1. Transform sampled CAD points into the checker frame.
2. For each observed object point, find the nearest transformed CAD point.
3. Reject correspondences with distance > icp_correspondence_dist_m.
4. Keep the best trimmed fraction of correspondences.
5. Solve the least-squares rigid alignment:

   minimize sum || R a_i + t - b_i ||^2

   using SVD of the cross-covariance matrix H = A_centered^T B_centered.

6. Compute:
   H = U Σ V^T
   R = V U^T
   if det(R) < 0, flip the last singular vector and recompute R
   t = b_mean - R a_mean

7. Form delta transform ΔT = [R t; 0 1].
8. Update the pose:
   T_checker_object ← ΔT T_checker_object

This is not Jacobian-based ICP. The SVD step does not assume small rotation for fixed correspondences, but ICP still needs a decent initial pose because nearest-neighbor correspondences can be wrong if the initial pose is too far away.
```

This is exactly the math behind your `rigid_transform_from_correspondences()` and `icp_refine_candidate()` flow.
