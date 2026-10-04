---
layout: post
title: Hessian Degeneracy Test
date: 2026-05-14 13:19
subtitle:
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine-Learning
---
# Metric 1 - Degeneracy using Hessian

## Part 1 - The derivation

Suppose the state is

$$
\mathbf{x} =
\begin{bmatrix}
\mathbf{p} \\
\mathbf{v} \\
\boldsymbol{\theta} \\
\mathbf{b}_a \\
\mathbf{b}_g \\
\mathbf{g}
\end{bmatrix}
\in \mathbb{R}^{18}.
$$

Each component is 3-dimensional, so the full state has 18 variables.

For each correspondence or measurement $i$, define a residual $\mathbf{r}_i(\mathbf{x})$. Its dimension $m$ depends on the registration method:

- **Point-to-point ICP:** $\mathbf{r}_i \in \mathbb{R}^3$
- **Point-to-plane ICP:** $r_i \in \mathbb{R}$
- **Point-to-line ICP:** often $r_i \in \mathbb{R}$, depending on the formulation
- **NDT:** can be represented with a 3D residual, $\mathbf{r}_i \in \mathbb{R}^3$, with the covariance used as a weighting or whitening term

For least-squares optimization, the cost of one measurement is (for simplicity we omit the information matrix $\Omega$)

$$
E_i = \frac{1}{2}\mathbf{r}_i^T\mathbf{r}_i.
$$

The total cost is

$$
E = \sum_i E_i = \frac{1}{2}\sum_i \mathbf{r}_i^T\mathbf{r}_i.
$$

Since $E_i$ is a scalar, its derivative with respect to the state is an $18\times1$ gradient.

Define the residual Jacobian as

$$
\mathbf{J}_i = \frac{\partial \mathbf{r}_i}{\partial \mathbf{x}}.
$$

If the residual has dimension $m$, then

$$
\mathbf{r}_i \in \mathbb{R}^{m},
\qquad
\mathbf{J}_i \in \mathbb{R}^{m\times18}.
$$

- Point-to-plane ICP has $m=1$, so $\mathbf{J}_i \in \mathbb{R}^{1\times18}$.
- Point-to-point ICP has $m=3$, so $\mathbf{J}_i \in \mathbb{R}^{3\times18}$.

Using the chain rule,

$$
\frac{\partial E_i}{\partial \mathbf{x}} = \mathbf{J}_i^T\mathbf{r}_i.
$$

After stacking all residuals into one vector,

$$
\mathbf{r} =
\begin{bmatrix}
\mathbf{r}_0 \\
\mathbf{r}_1 \\
\vdots
\end{bmatrix},
$$

and all residual Jacobians into one matrix,

$$
\mathbf{J} =
\begin{bmatrix}
\mathbf{J}_0 \\
\mathbf{J}_1 \\
\vdots
\end{bmatrix},
$$

the total cost becomes

$$
E = \frac{1}{2}\mathbf{r}^T\mathbf{r},
$$

and its gradient is

$$
\nabla E = \mathbf{J}^T\mathbf{r}.
$$

The Hessian is the derivative of the gradient:

$$
\nabla^2 E = \frac{\partial^2 E}{\partial \mathbf{x}^2}.
$$

For a scalar residual $r_i$, the gradient of $E_i$ is

$$
\nabla E_i = r_i \nabla r_i.
$$

Differentiating once more with the product rule gives

$$
\nabla^2 E_i = (\nabla r_i)(\nabla r_i)^T + r_i\nabla^2 r_i.
$$

Since $\nabla r_i = \mathbf{J}_i^T$, we obtain

$$
\nabla^2 E = \sum_i \left( \mathbf{J}_i^T\mathbf{J}_i + r_i\nabla^2 r_i \right).
$$

For a vector residual, the same idea applies component-wise:

$$
\nabla^2 E = \mathbf{J}^T\mathbf{J} + \sum_k r_k \nabla^2 r_k.
$$

The first term, $\mathbf{J}^T\mathbf{J}$, depends only on the first derivatives of the residuals. The second term depends on their second derivatives and is weighted by the residual values themselves.

Near a good solution, the residuals $r_k$ are expected to be small. Gauss-Newton therefore neglects the second term and approximates the Hessian as

$$
\boxed{\mathbf{H} \approx \mathbf{J}^T\mathbf{J}}
$$

For an 18-dimensional state,

$$
\mathbf{J}^T\mathbf{J} \in \mathbb{R}^{18\times18}.
$$

This approximate Hessian describes how strongly the registration cost changes for different perturbations of the state.

## Part 2 - Why Hessian eigenvalues can show degeneracy

Suppose we perturb the state $\mathbf{x}$ by a small amount $\delta\mathbf{x}$. A second-order Taylor expansion of the cost is

$$
E(\mathbf{x}+\delta\mathbf{x})
\approx
E(\mathbf{x})
+ \nabla E(\mathbf{x})^T\delta\mathbf{x}
+ \frac{1}{2}\delta\mathbf{x}^T \mathbf{H}\, \delta\mathbf{x}.
$$

Near a local optimum,

$$
\nabla E(\mathbf{x}) \approx 0,
$$

so the change in cost is approximately

$$
E(\mathbf{x}+\delta\mathbf{x}) - E(\mathbf{x})
\approx
\frac{1}{2}\delta\mathbf{x}^T \mathbf{H}\, \delta\mathbf{x}.
$$


Now consider the eigendecomposition of the Hessian:

$$
\mathbf{H}\mathbf{v}_i = \lambda_i\mathbf{v}_i.
$$

Because the **Hessian is symmetric**, an $18\times18$ Hessian has 18 eigenvalues and we can choose 18 orthonormal eigenvectors,

$$
\mathbf{v}_1,\ldots,\mathbf{v}_{18},
$$

which form a basis of the 18-dimensional error-state space.

Any state perturbation can therefore be written as

$$
\delta\mathbf{x} = \sum_{i=1}^{18} a_i\mathbf{v}_i.
$$

Substituting this into the quadratic form gives

$$
\delta\mathbf{x}^T \mathbf{H}\, \delta\mathbf{x} = \sum_{i=1}^{18} \lambda_i a_i^2.
$$

Therefore,

$$
\boxed{
\Delta E \approx \frac{1}{2}\sum_{i=1}^{18} \lambda_i a_i^2
}
$$

### Small eigenvalues

If an eigenvalue $\lambda_i$ is very small, then a perturbation along its eigenvector $\mathbf{v}_i$ changes the cost only slightly. For example, let

$$
\delta\mathbf{x} = \alpha\mathbf{v}_i.
$$

Then

$$
\Delta E \approx \frac{1}{2}\alpha^2\lambda_i.
$$

If

$$
\lambda_i \approx 0,
$$

then

$$
\Delta E \approx 0
$$

even for a nonzero perturbation $\alpha$. This means that changing the state along that direction produces almost no change in the registration cost. The measurements therefore provide little information about that state direction.

Here, a "direction" does not necessarily mean a physical direction in 3D space. The perturbation

$$
\delta\mathbf{x}\in\mathbb{R}^{18}
$$

lives in the error-state space. An eigenvector might represent pure translation, pure rotation, or a combination of multiple state variables. For example, an eigenvector could look schematically like

$$
\begin{bmatrix}
0.7 & 0.7 & 0 & 0 & \cdots & 0
\end{bmatrix}^T,
$$

meaning that the poorly observed direction is a combination of two state variables rather than a single coordinate axis.

A small eigenvalue also makes the solver unstable: the update $\delta\mathbf{x} = -\mathbf{H}^{-1}\mathbf{b}$ divides by $\lambda_i$ along $\mathbf{v}_i$, so noise in that direction gets amplified. See [zero eigenvalues, invertibility, and condition number](https://ricojia.github.io/2017/01/15/eigen-value-decomp/#zero-eigenvalues-invertibility-and-condition-number).

### Large eigenvalues

Conversely, suppose every eigenvalue is bounded away from zero:

$$
\lambda_i \ge \lambda_{\min} > 0.
$$

Then

$$
\sum_i \lambda_i a_i^2 \ge \lambda_{\min}\sum_i a_i^2.
$$

Because the eigenvectors are orthonormal,

$$
\sum_i a_i^2 = \|\delta\mathbf{x}\|^2.
$$

Therefore,

$$
\boxed{
\Delta E \gtrsim \frac{1}{2}\lambda_{\min}\|\delta\mathbf{x}\|^2
}
$$

So if the smallest eigenvalue is sufficiently far from zero, there is no nonzero unit state direction that can be changed without noticeably increasing the cost. In that sense, there is no locally unobservable direction in the Hessian.


"Small" or "large" eigenvalues are relative quantities. In an 18-state LIO system, different state components have different units, such as meters, radians, meters per second, and sensor-bias units. For this reason, **LiDAR geometric degeneracy is often analyzed using the $6\times6$ pose Hessian associated with translation and rotation rather than the full $18\times18$ estimator Hessian.**

## Plane is a Natural Degeneracy Scenario

Some point-cloud geometries naturally provide weak constraints on certain pose directions. A common example is a single plane.

![](https://i.postimg.cc/d1k0kWYP/chuan-gan-qi-yu-qiang-mian-fa-xiang-shi-yi-tu.png)

Imagine the robot facing a large flat wall. In point-to-plane ICP, each residual measures 
displacement along the wall normal. If the robot moves parallel to the wall, the point-to-plane residuals change very little. Therefore,

$$  
\mathbf{J}\delta\mathbf{x} \approx 0,  
$$

and consequently

$$  
\Delta E  
\approx  
\frac{1}{2}  
\delta\mathbf{x}^T  
\mathbf{H}  
\delta\mathbf{x}  
\approx 0.  
$$

These weakly constrained motions appear as small eigenvalues of the pose Hessian.

For an ideal infinite plane, two translations along the plane are unobservable. Rotation about the plane normal is also unobservable in pure point-to-plane ICP. Therefore, an ideal single-plane geometry can produce three near-zero eigenvalues in the $6\times6$ pose Hessian.

Other common degeneracy cases include long corridors or tunnels, line-like structures, scenes dominated by parallel planes, narrow fields of view, and highly repetitive or symmetric geometry. In each case, the underlying issue is the same: some pose perturbation changes the measured geometric residuals very little.


---

## Part 3 - What To Do For Degeneracy

The core idea of degeneracy handling is to **update the state only along well-constrained directions**, and leave weakly constrained directions alone.

### Why the plain Gauss-Newton step is unstable

[Recall that](https://ricojia.github.io/2024/07/11/rgbd-slam-bundle-adjustment/#how-to-solve-for-delta-x) Gauss-Newton solves for the step from the normal equations, with the gradient $\mathbf{b} = \mathbf{J}^T\mathbf{r}$:

$$
\mathbf{H}\,\delta\mathbf{x} = -\mathbf{b}
\quad\Rightarrow\quad
\delta\mathbf{x} = -\mathbf{H}^{-1}\mathbf{b}
$$

Write the eigendecomposition of the Hessian, **with eigenvalues sorted from largest to smallest**, and expand the gradient in the eigenvector basis:

$$
\begin{gather*}
\mathbf{H} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T,
\quad
\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \ldots, \lambda_n),
\quad
\lambda_1 \ge \cdots \ge \lambda_n
\\
\mathbf{b} = \sum_i \beta_i \mathbf{v}_i,
\quad
\beta_i = \mathbf{v}_i^T\mathbf{b}
\end{gather*}
$$

Then the step is

$$
\delta\mathbf{x} = -\sum_{i=1}^{n} \frac{\beta_i}{\lambda_i}\mathbf{v}_i
$$

Along a weak direction, $\lambda_i \approx 0$, so even a small $\beta_i$ (which may be pure measurement noise) produces a huge step. [A worked numerical example is here](https://ricojia.github.io/2017/01/15/eigen-value-decomp/#zero-eigenvalues-invertibility-and-condition-number).

### Step 1: Find the weak directions

Pick an eigenvalue threshold $\lambda_{th}$. Directions with $\lambda_i < \lambda_{th}$ are treated as degenerate. Build a diagonal mask $\mathbf{D}$:

$$
\mathbf{D} = \operatorname{diag}(d_1, \ldots, d_n),
\quad
d_i =
\begin{cases}
1 & \lambda_i \ge \lambda_{th} \\
0 & \lambda_i < \lambda_{th}
\end{cases}
$$

For the single-plane example above, the $6\times6$ pose Hessian has three near-zero eigenvalues, so $\mathbf{D} = \operatorname{diag}(1, 1, 1, 0, 0, 0)$.

The threshold is on the eigenvalue $\lambda_i$, which measures how well the measurements constrain a direction. 

Now we would like a projection matrix $\mathbf{P}$ that keeps only the well-constrained directions. That is, for each eigenvector:

$$
\mathbf{P}\mathbf{v}_i =
\begin{cases}
\mathbf{v}_i & \lambda_i \ge \lambda_{th} \text{ (well constrained: keep)} \\
\mathbf{0} & \lambda_i < \lambda_{th} \text{ (weak: remove)}
\end{cases}
\quad\Longleftrightarrow\quad
\mathbf{P}\mathbf{v}_i = d_i\mathbf{v}_i
$$

So $\mathbf{P}$ should have the same eigenvectors as $\mathbf{H}$, with eigenvalues $d_i$ instead of $\lambda_i$. Swapping $\boldsymbol{\Lambda}$ for $\mathbf{D}$ in the eigendecomposition gives exactly that:

$$
\mathbf{P} = \mathbf{V}\mathbf{D}\mathbf{V}^T
$$

To check, the eigenvectors are orthonormal, so $\mathbf{V}^T\mathbf{v}_i = \mathbf{e}_i$, the $i$-th standard basis vector. Then

$$
\mathbf{P}\mathbf{v}_i = \mathbf{V}\mathbf{D}\mathbf{V}^T\mathbf{v}_i = \mathbf{V}\mathbf{D}\mathbf{e}_i = d_i\mathbf{V}\mathbf{e}_i = d_i\mathbf{v}_i
$$

Since any vector is a combination of the $\mathbf{v}_i$, $\mathbf{P}$ keeps its components along well-constrained eigenvectors and zeroes the rest. $\mathbf{P}$ is also symmetric, and idempotent ($\mathbf{P}^2 = \mathbf{V}\mathbf{D}^2\mathbf{V}^T = \mathbf{P}$, since $d_i^2 = d_i$): projecting twice changes nothing.

### Step 2: Solve only in the well-constrained subspace

There are three ways to think about the projected step, and they give the same answer.

**Option A: solve, then project** (solution remapping, as in Zhang, Kaess, and Singh, "On Degeneracy of Optimization-based State Estimation Problems", ICRA 2016):

$$
\delta\mathbf{x}' = \mathbf{P}\,\delta\mathbf{x}
$$

The huge components of $\delta\mathbf{x}$ lie exactly along the weak eigenvectors, so $\mathbf{P}$ removes them. **The caveat is that this still requires $\mathbf{H}^{-1}$ to exist, so it fails if some $\lambda_i$ is exactly zero.**

**Option B: project the problem, then solve.** Restrict the step to $\delta\mathbf{x} = \mathbf{P}\mathbf{y}$ and substitute into the quadratic cost:

$$
\min_{\mathbf{y}}\ \frac{1}{2}(\mathbf{P}\mathbf{y})^T\mathbf{H}(\mathbf{P}\mathbf{y}) + \mathbf{b}^T(\mathbf{P}\mathbf{y})
$$

This expression is the cost we minimize, not an equation, so it is not set to zero. The minimum is where its **gradient** with respect to $\mathbf{y}$ is zero:

$$
\mathbf{P}^T\mathbf{H}\mathbf{P}\,\mathbf{y} + \mathbf{P}^T\mathbf{b} = \mathbf{0}
$$

Because $\mathbf{P}$ is symmetric ($\mathbf{P}^T = \mathbf{P}$), the normal equations become

$$
\mathbf{H}'\mathbf{y} = -\mathbf{b}',
\quad
\mathbf{H}' = \mathbf{P}\mathbf{H}\mathbf{P},
\quad
\mathbf{b}' = \mathbf{P}\mathbf{b}
$$

However, $\mathbf{H}' = \mathbf{V}(\mathbf{D}\boldsymbol{\Lambda}\mathbf{D})\mathbf{V}^T$ has exact zeros on the masked directions, so **$\mathbf{H}'$ is singular and cannot be inverted directly**. Solve it with the pseudo-inverse, which inverts only the nonzero eigenvalues: $\delta\mathbf{x}' = -(\mathbf{H}')^{+}\mathbf{b}'$.

**Option C: truncated eigen-solution.** Drop the weak terms from the sum directly:

$$
\boxed{
\delta\mathbf{x}' = -\sum_{i:\ \lambda_i \ge \lambda_{th}} \frac{\beta_i}{\lambda_i}\mathbf{v}_i
}
$$

Expanding Options A and B in the eigenvector basis gives exactly this sum. **Option C is my favorite because it never divides by a small eigenvalue**, so it is numerically safe even when some $\lambda_i = 0$. It is the eigenvalue analogue of a truncated SVD.

### In LIO: apply it to the LiDAR term only

In LIO, the Hessian also includes a prior from IMU propagation:

$$
\mathbf{H} = \mathbf{H}_{prior} + \mathbf{H}_{lidar},
\qquad
\mathbf{b} = \mathbf{b}_{prior} + \mathbf{b}_{lidar}
$$

Degeneracy handling should only be applied to $\mathbf{H}_{lidar}$, for two reasons:

1. **Detection:** the prior makes the full $\mathbf{H}$ well-conditioned, which hides LiDAR degeneracy. Run the eigenvalue test on the $6\times6$ pose block of $\mathbf{H}_{lidar}$.
2. **Correction:** along a weak direction, we want the estimate to follow the prior rather than the LiDAR. So project only the LiDAR contribution:

$$
\mathbf{H}' = \mathbf{H}_{prior} + \mathbf{P}\mathbf{H}_{lidar}\mathbf{P},
\qquad
\mathbf{b}' = \mathbf{b}_{prior} + \mathbf{P}\mathbf{b}_{lidar}
$$

Here $\mathbf{P}$ is built from the pose block and padded with identity on the other states. Unlike Option B, $\mathbf{H}'$ stays full rank because $\mathbf{H}_{prior}$ fills in the masked directions, so $\delta\mathbf{x}' = -\mathbf{H}'^{-1}\mathbf{b}'$ can be solved normally.

---

## Part 4 - Additional Handling For Rotation Scale

### Why one threshold does not fit both units

Part 3 flags a direction as weak when its eigenvalue is small relative to the largest one:

$$
\frac{\lambda_i}{\lambda_{\max}} < \tau
$$

For the $6\times6$ pose Hessian, the perturbation

$$
\delta\mathbf{x} =
\begin{bmatrix}
\delta\theta_x & \delta\theta_y & \delta\theta_z & \delta t_x & \delta t_y & \delta t_z
\end{bmatrix}^T
$$

mixes radians and meters, so its eigenvalues are not in the same units either. Comparing them against one threshold $\tau$ is not meaningful.

### How to handle that

A naive thought is

 λ_i / λ_max < scale_i * THRESHOLD

This doesn't work well because scaling the threshold per eigenvalue breaks on coupled directions. Suppose an eigenvector is 60% yaw and 40% sideways motion, another is 20% yaw and 40% sideways motion. We cannot apply a single preset scale_i. 

Instead, we can convert radians into meters with a characteristic length $L$:

$$
\delta s \approx L\,\delta\theta
$$

If most points are about $L$ meters away, a rotational perturbation $\delta\theta$ moves them by about $\delta s$ meters. So we introduce a scaled coordinate in which every component is in meters:

$$
\delta\mathbf{x}_s = \mathbf{S}\,\delta\mathbf{x},
\qquad
\mathbf{S} = \operatorname{diag}(L, L, L, 1, 1, 1)
$$

This $\mathbf{S}$ is for the rotation-first ordering above. For the translation-first ordering $[\delta\mathbf{p}, \delta\boldsymbol{\theta}]$ used in Part 1, it is $\operatorname{diag}(1, 1, 1, L, L, L)$.

Substituting $\delta\mathbf{x} = \mathbf{S}^{-1}\delta\mathbf{x}_s$ into the quadratic cost:

$$
\Delta E
= \frac{1}{2}\delta\mathbf{x}^T\mathbf{H}\,\delta\mathbf{x}
= \frac{1}{2}(\mathbf{S}^{-1}\delta\mathbf{x}_s)^T\mathbf{H}(\mathbf{S}^{-1}\delta\mathbf{x}_s)
= \frac{1}{2}\delta\mathbf{x}_s^T\left(\mathbf{S}^{-T}\mathbf{H}\mathbf{S}^{-1}\right)\delta\mathbf{x}_s
$$

So the Hessian whose eigenvalues should be compared is

$$
\boxed{
\mathbf{H}_s = \mathbf{S}^{-T}\mathbf{H}\mathbf{S}^{-1}
}
$$

Here the eigenvectors belong to $\mathbf{H}_s$, so the projector $\mathbf{P}_s = \mathbf{V}_s\mathbf{D}\mathbf{V}_s^T$ acts on scaled coordinates. To apply it to a real step, we now need to convert back:

$$
\delta\mathbf{x}' = \mathbf{S}^{-1}\mathbf{P}_s\mathbf{S}\,\delta\mathbf{x}
\quad\Rightarrow\quad
\mathbf{P} = \mathbf{S}^{-1}\mathbf{P}_s\mathbf{S}
$$

This $\mathbf{P}$ is still idempotent, but it is **no longer symmetric**. So the projected system must be written with $\mathbf{P}^T$ (transpose was omitted previously because P used to be symmetric):

$$
\mathbf{H}' = \mathbf{P}^T\mathbf{H}\mathbf{P},
\qquad
\mathbf{b}' = \mathbf{P}^T\mathbf{b}
$$
### Choosing $L$ per scan

**Idea:** $L$ is the lever arm. A lever arm is the distance from the pivot to where something moves: push a door near its hinge and the handle barely moves, but the far edge swings a lot. The same holds when the sensor rotates. The sensor is the pivot, and turning it by a small angle $\delta\theta$ moves a point $r$ meters away by about $r\,\delta\theta$. Far points swing a lot and near points barely move. So $L$ should be the "typical" range of the scan, and since that changes from scan to scan (2-5 m in a corridor, 30 m+ outdoors), we compute it from each scan.

Let $\mathbf{p}_i$ be a matched point in the body (IMU) frame, and $N$ the number of matched points.

**Step 1: what each point adds to the Hessian.** For point-to-point ICP, the residual is $\mathbf{r}_i = \mathbf{R}\mathbf{p}_i + \mathbf{t} - \mathbf{q}_i$, with $\mathbf{q}_i$ the matched map point. Near the current estimate, each point contributes to the Hessian block by:

|Perturbation|Residual change|Jacobian|Hessian block $\mathbf{J}^T\mathbf{J}$|
|---|---|---|---|
|Translate by $\delta\mathbf{t}$|$\delta\mathbf{t}$|$\mathbf{I}$|$\mathbf{I}$|
|Rotate by $\delta\boldsymbol{\theta}$|$\delta\boldsymbol{\theta}\times\mathbf{p}_i = -[\mathbf{p}_i]_\times\delta\boldsymbol{\theta}$|$-[\mathbf{p}_i]_\times$|$[\mathbf{p}_i]_\times^T[\mathbf{p}_i]_\times$|

**Step 2: measure how "stiff" each block is.** Think of the cost as a spring holding the pose in place: $\Delta E \approx \frac{1}{2}\delta\mathbf{x}^T\mathbf{H}\,\delta\mathbf{x}$, so pushing the pose by $\delta\mathbf{x}$ raises the cost like stretching a spring. A large eigenvalue is a stiff spring (a small push costs a lot, so that direction is well constrained), and a small eigenvalue is a loose one. The trace (sum of the diagonal, which equals the sum of the eigenvalues) adds up the stiffness of a block's three directions, so it measures the block's total stiffness. 

Using $[\mathbf{p}]_\times^T[\mathbf{p}]_\times = \|\mathbf{p}\|^2\mathbf{I} - \mathbf{p}\mathbf{p}^T$, whose trace is $3\|\mathbf{p}\|^2 - \|\mathbf{p}\|^2 = 2\|\mathbf{p}\|^2$:

$$
\operatorname{tr}(\mathbf{H}_{tt}) = 3N,
\qquad
\operatorname{tr}(\mathbf{H}_{\theta\theta}) = 2\sum_i \|\mathbf{p}_i\|^2
$$

Why does the rotation block grow with the *squared* range? A rotation $\delta\theta$ moves a point at range $r$ by $r\,\delta\theta$, so its residual changes by about $r\,\delta\theta$. The cost is the residual squared, so it rises by about $r^2\,\delta\theta^2$. Translation moves every point by the same $\delta t$, whatever its range, so its block has no $r$ in it. That $r^2$ is the lever arm showing up in the Hessian.

**Step 3: pick $L$ so the two blocks match.** The relative test $\lambda_i / \lambda_{\max} < \tau$ only works if rotation and translation start on an equal footing. Otherwise the stiffer block supplies $\lambda_{\max}$, and everything in the other block looks weak. So we choose $L$ to make the two blocks equally stiff on average.

Scaling divides the rotation block by exactly $L^2$ ($\mathbf{S}$ is diagonal, so $\mathbf{H}_{\theta\theta}$ becomes $\mathbf{H}_{\theta\theta}/L^2$). Each block is $3\times3$, so its average eigenvalue is its trace divided by 3. Equal averages therefore means equal traces:

$$
\frac{\operatorname{tr}(\mathbf{H}_{\theta\theta})}{L^2} = \operatorname{tr}(\mathbf{H}_{tt})
\quad\Rightarrow\quad
L^2 = \frac{2\sum_i \|\mathbf{p}_i\|^2}{3N}
$$

The trace is used rather than, say, the largest eigenvalue, because it is a plain sum over points, which gives the closed form below.

$$
\boxed{
L = \sqrt{\frac{2}{3}\cdot\frac{1}{N}\sum_i \|\mathbf{p}_i\|^2}
\approx 0.82 \times \text{RMS range}
}
$$

It is the RMS (root-mean-square) range rather than the plain average, because the Hessian grows with range squared, so far points count more.

**Practical notes:**

- **Matched points only.** Unmatched points don't contribute to $\mathbf{H}$.
- **Body frame.** The Jacobian $-\mathbf{R}[\mathbf{p}_{imu}]_\times$ rotates about the IMU, so measure ranges from the IMU.
- **NDT or point-to-plane:** the per-point weights change both blocks, so the formula is approximate, but the RMS range is still a sensible scale.
- **Clamp and smooth.** Put a floor on $L$ for near-empty scans, and smooth it across scans (e.g. an exponential moving average) so the degeneracy decision doesn't flicker.

### Related work 

Making rotation and translation comparable with a length scale is an established idea. 
1. In robot kinematics, Angeles and López-Cajún (1992) introduced a *characteristic length* that makes a manipulator Jacobian dimensionally homogeneous, so that its condition number is meaningful. 
2. For LiDAR degeneracy, [DCReg](https://arxiv.org/abs/2509.06285) (2025) analyzes this rotation-translation scale disparity and decouples the two subspaces with Schur complements.
3. [Degeneracy-Resilient Teach and Repeat with FMCW Lidar](https://arxiv.org/abs/2603.10248) (2026) uses block scaling with a scaling factor $\ell$, detects degeneracy in the scaled space, and maps the solution back, as in this post.
