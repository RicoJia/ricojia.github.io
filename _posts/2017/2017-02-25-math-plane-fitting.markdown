---
layout: post
title: Math - Linear Fitting
date: '2017-02-25 13:19'
subtitle: Plane Fitting, Line Fitting
comments: true
tags:
    - Math
---

Given a group of points `x_i = [x, y, z]`, how do we find a plane and a line that best fit them?

## Plane Fitting

Points on a plane satisfy:

$$
\begin{gather*}
\begin{aligned}
& n^Tx + d = 0
\end{aligned}
\end{gather*}
$$

Of course, in reality there very likely exist points in this group that don't satisfy this condition. However, we can choose `nx+d` as an "error" and minimize the total error across the point group. This problem becomes solving a linear least-squares problem:

$$
\begin{gather*}
\begin{aligned}
& min \sum_{k=1}^n |n^T x_k + d|^2
\end{aligned}
\end{gather*}
$$

If we stack `x_k` together, and introduce $\tilde{n}$

$$
\begin{gather*}
\begin{aligned}
& X = \begin{bmatrix}
x_1, 1 \\
x_2, 1 \\
\cdots
\end{bmatrix}

& \tilde{n} = \begin{bmatrix}
n_x \\
n_y \\
n_z \\
d
\end{bmatrix}
\end{aligned}
\end{gather*}
$$

X is overdetermined. Also, since we are interested in the direction $[n_x, n_y, n_z]$, we can normalize $\tilde{n}$, which normalizes d as well but would not affect the result of the fitting. Then we can write the above as:

$$
\begin{gather*}
\begin{aligned}
& min_{n=1}^k |X \tilde{n}|^2
\end{aligned}
\end{gather*}
$$

Because X is a simple matrix, this is a simple linear least-square problem. To solve it, one can use Eigen Value Decomposition or Singular Value Decomposition.

### Eigen Decomposition

For eigen decomposition,

$$
\begin{gather*}
\begin{aligned}
& |X \tilde{n}|^2 = n^T X^T X n
\\ & \text{since: } X = V \Lambda V^{-1}
\\ & \Rightarrow
|X \tilde{n}|^2 = n^T  V \Lambda V^{-1} n
\end{aligned}
\end{gather*}
$$

Here, we pack eigen values and eigen vectors of $X$ into: $\Lambda = diag(\lambda_1^2, \lambda_2^2, \cdots)$. $V = [v_1, v_2 ..., v_n]$. 

In the meantime, since V is a vector basis in $R^n$, we can represent $n = \alpha_1 v_1 + \cdots \alpha_n v_n$:

After plugging the above into $n^T  V \Lambda V^{-1} n$, we see that:

$$
\begin{gather*}
\begin{aligned}
& |X \tilde{n}|^2 = n^T  V \Lambda V^{-1} n
\\ & = \lambda_1 \alpha_1^2 + ... + \lambda_k \alpha_k^2
\end{aligned}
\end{gather*}
$$

Because $\newcommand{\abs}[1]{\left| #1 \right|} \abs{\tilde{n}} = 1$, assuming eigen values are in descending order through $\lambda_1 \cdots \lambda_k$:

$$
\begin{gather*}
\begin{aligned}
& argmin_{\tilde{n}} |X \tilde{n}|^2 = v_1
\end{aligned}
\end{gather*}
$$

This corresponds to $a_1 = 1$, $a_2 = \cdots = a_n = 0$. The total error is minimized.

### Singular Value Decomposition

Equivalently, we can use Singular Value Decomposition:

$$
\begin{gather*}
\begin{aligned}
& X = U \Sigma V^T
\end{aligned}
\end{gather*}
$$

Where `U` and `V` are `mxm, nxn` singular vectors, which are also orthonormal basis. Specifically, `V` is the eigen basis of `X`. $Sigma = diag(\lambda_1, \lambda_2, \cdots)$, which is a mxn diagonal matrix with eigen values of $X^TX$. So the above gives us:

$$
\begin{gather*}
\begin{aligned}
& |X \tilde{n}|^2 = n^T X^T X n
\\ & = V \Sigma^2 V^T
\end{aligned}
\end{gather*}
$$

This is equivalent to eigen decomposition.

### Why The Above Doesn't Need Gaussian Newton

Recall that in [robot pose estimation](https://ricojia.github.io/2024/07/11/rgbd-slam-bundle-adjustment/#how-to-formulate-slam-into-an-optimization-problem), cost is also in the quadratic form:

$$
\begin{gather*}
\begin{aligned}
& F_{ij} = e_{ij} \Omega e_{ij}
\end{aligned}
\end{gather*}
$$

But the above needs Gauss Newton because $e_{ij}$ is non linear w.r.t the pose of the robot. In a 2D pose estimation scenario, it would be:

$$
\begin{gather*}
\begin{aligned}
& e_{ij} =
\begin{bmatrix}
x_j - (x_i + d_i*cos(\theta_i + \psi_i)), \\
y_j - (y_i + d_i*sin(\theta_i + \psi_i))
\end{bmatrix}
\end{aligned}
\end{gather*}
$$

Gauss-Newton iteratively linearizes the neighbor landscape of the cost $F$ so it can estimate the cost jacobian, which gives the minimum cost, w.r.t the pose variables.

In plane fitting, $X$ is a single matrix, which makes it a linear-least-square minimization. In that case, we don't need to iteratively linearize the cost landscape. So Gauss Newton is not needed.

## Line Fitting

In linear fitting, a plane is a bit easier because a plane needs 4 variables, with 1 constraint. A plane needs 6 variables. A line can be represented as:

$$
\begin{gather*}
\begin{aligned}
& x = p_0 + dt
\end{aligned}
\end{gather*}
$$

`t` here is a variable. We want to find 2 variables: $p_0$ and $d$, either is a 3x1 vector. To find the best line parameters, we find the minimum total distance from points to the line. By linking a point to $p_0$, we can use the Pythogorean Theorem to solve it:

$$
\begin{gather*}
\begin{aligned}
& argmin \sum_k |(x_k-p_0)^2 - (d^T (x_k - p_0))^2 = argmin \sum_k f_k^2
\\ & \text{subject to: }
|d| = 1
\end{aligned}
\end{gather*}
$$

For a single point, the partial derivative of the cost w.r.t parameters is:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial f_k^2}{\partial p_0} = -2(\mathbf{x}_k - \mathbf{p_0}) + 2 \left( (\mathbf{x}_k - \mathbf{p_0})^\top \mathbf{d} \right) \mathbf{d},

\\ & \text{Scalar =>} = \mathbf{d}^\top (\mathbf{x}_k - \mathbf{p_0})
\\ & = (-2)(I - \mathbf{d} \mathbf{d}^\top)(\mathbf{x}_k - \mathbf{p_0}).
\end{aligned}
\end{gather*}
$$

After summation:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial \sum_{k=1}^{n} f_k^2}{\partial p} = \sum_{k=1}^{n} (-2)(I - \mathbf{d} \mathbf{d}^\top)(\mathbf{x}_k - \mathbf{p}),

\\&
= (-2)(I - \mathbf{d} \mathbf{d}^\top) \sum_{k=1}^{n} (\mathbf{x}_k - \mathbf{p}).
\end{aligned}
\end{gather*}
$$

We can make it zero by having $p_0$ be the center of the point cloud:

$$
\begin{gather*}
\begin{aligned}
& p_0 = \frac{1}{n} \sum_{k} x_k
\end{aligned}
\end{gather*}
$$

**With known $p_0$**, we can now find $d$. Let $y_k = x_k - p$:

$$
\begin{gather*}
\begin{aligned}
& f_k^2 = y_k^T y_k - d^T y_k y_k^T d
\\ & \Rightarrow d^* = argmax \sum_k d^T y_k y_k^T d = \sum_k |y_k^T d| ^2
\end{aligned}
\end{gather*}
$$

We can stack $y_k$ together:

$$
\begin{gather*}
\begin{aligned}
& Y = \begin{bmatrix}
y_1^T
\\ \cdots
\\ y_k^T
\end{bmatrix}
\end{aligned}
\end{gather*}
$$

$$
\begin{gather*}
\begin{aligned}
& d^* = argmax |Yd|^2
\end{aligned}
\end{gather*}
$$

Then, we can solve this with eigen decomposition!

From the perspective of SVD, we can find that the line is the first principal component. With the second principal compomnet, we can find a plane. The plane's normal vector is the smallest principal component. $A^T A$ is the **covariance matrix**.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/e61e0bb2-28cc-4217-bde9-d7038d897e82" height="300" alt=""/>
       </figure>
    </p>
</div>

Great thing about Eigen Value Decomposition is that we do not need to iteratively evaluate.
