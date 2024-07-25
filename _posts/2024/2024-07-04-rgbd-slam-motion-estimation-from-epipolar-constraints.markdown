---
layout: post
title: Motion Estimation From Epipolar Constraints  
date: '2024-07-04 10:11'
excerpt: This is an Introduction of Epipolar Constraints and 2D-2D Motion Estimation
comments: true
---

## Intro

This blog is inspired by this great great book: [14 Lectures in Visual SLAM](https://link.springer.com/book/10.1007/978-981-16-4939-4), authored by Dr. Gao Xiang et al. This article will walk you through **8-point algorithm** when features lie on different planes, and Direct Linear Transforms when all features are on the same plane.

## Theoretical Thingies

<p align="center">
<img src="https://github.com/RicoJia/Omnid_Project/assets/39393023/e6f684d8-de6c-4185-af21-f878ae7d5b33" height="300" width="width"/>
</p>

### Relative Motion Is In Epipolar Constraints

In the Epipolar Geometry show, our main characters are:
- $O_1$,  $O_2$ are the optical centers of two camera poses. Each one of them has a camera coordinate frame attached to them.
- $p_1$, $p_2$ are the corresponding pixel points
- $P$ is the 3D point 
- Additionally, $e_1$, $e_2$ are the epipoles of the two cameras. $l_1$ and $l_2$ are epipolar lines
- $O_1 O_2$ is called the baseline

The main purpose of this show is to establish a constraint between the pixels, using the co-planar characteristic of $O_1$, $P$, and $O_2$

Below we are denoting 3D points using the capital letter $P$, and 2D pixel points using lower case letter $p$ . Let $P_1=[X,Y,Z]$ in $O_1$. Assume there's transformation $O_2 = RO_1 + T$. Then, the 3D point $P_2 = RP_1 + t$, where the **translation vector t** is $\vec{O_1O_2}$

## Step 1 - Epipolar Constraints Derivation

In the pinhole camera model, we introduce the notion of "canonical plane". Remember that our image plane is focal length $f$ away from the optical center? the canonical plane is 1 unit away from it. Meanwhile, the point $P$ has depth $Z_1$ and $Z_2$ in each frame. This simplifies our computation quite a bit. We represent points on the canonical planes of the two cameras as $P_{c1}$, $P_{c2}$

From the pinhole camera model, we know:

$$
\begin{align*}
KP_1 = Z_1 p_1 \\
K(RP_1+t) = Z_2 p_2
\end{align*}
$$

Then, we "normalize" the depths $Z_1$ and $Z_2$ and get the canonical points:

$$
\begin{gather*}
P_{c1} = K^{-1}p_1 = P_1/Z_1 \\
P_{c2} = K^{-1}p_2 = (RP_1+t)/Z_2 = (Z_1 RP_{c1}+t)/Z_2
\end{gather*}
$$

### How do we define the epipolar constraint?

Using the coplanar characteristic, and the **essential matrix, E**:

$$
\begin{gather*}
t \times P_{c2} = \alpha t \times RP_1
\\
=>
\\
P_{c2}^T \cdot t \times P_{c2} = 0
\\
=>
\\ 
P_{c2}^T \cdot t \times RP_{c1} = P_{c2}^T E P_{c1} = 0
\\
=>
E = t \times R
\end{gather*}
$$

One important point is **we omitted depth** because one side is zero. So, **the epipolar constraint is scale ambiguous**.

Note that we can use the skew matrix of $t$ to represent $E$ as $E = [t_\times] R$.

Now, to get the **foundamental matrix, F**,

$$
\begin{gather*}
P_{c2}^T E P_{c1} = 0
\\
=>
p_2^T K^{-T}FK^{-1} p_1  = p_2^T F p_1 = 0
\\
=>
F = K^{-T}EK^{-1}
\end{gather*}
$$

## Step 2 - Estimate Relative Motion in Epipolar Constraints (8 point algorithm)

In 1981, Longuet-Higgins proposed the famous "8-point algorithm" in Nature to estimate E, and solve for R and t. Since usually $K$ is known, we can operate on the canonical points

E is a 3x3 matrix:
$$
\begin{bmatrix}
e_1 & e_2 & e_3
\\
e_4 & e_5 & e_6
\\
e_7 & e_8 & e_9
\end{bmatrix}
$$

Let $e = [e_1 , e_2 , e_3 , e_4 , e_5 , e_6 , e_7 , e_8 , e_9 ]^T$

### ❓Remember how to get the canonical plane coordinates from pixel values?

Recall the intrinsic matrix

$$
\begin{gather*}
K = \begin{pmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{pmatrix}
\\
=>
\\
x_n = \frac{u - c_x}{f_x}, y_n = \frac{v - c_y}{f_y}
\end{gather*}
$$

### 🤔 How many equations do we need?

P_{c1} = $[u_1, v_1, 1]$ and P_{c2} is $[u_2, v_2, 1]$. 

So, each epipolar constraint $P_{c2}^T E P_{c1}$ gives 1 equation:

$$
\begin{gather*}
[u_2 u_1 , u_2 v_1 , u_2 , v_2 u_1 , v_2 v_1 , v_2 , u_1 , v_1 , 1] · e = Ae = 0
\end{gather*}
$$

Because the epipolar constraint is scale ambiguous, E multiplies any scalar would also be a valid essenstial matrix. So, we have 1 degree of freedom, hence we need 8 equations. Hence, we get 8 matched feature points, choose $e$ in the null space of $A$, voila!

### How to select those 8 points?

In general, there could be multiple outliers in matched feature pairs. But also in the mean time, we most likely get more than 8 feature pairs . So, in general, we:
1. Randomly select feature pairs that are within a specified distance from epipolar lines.
2. Calculate E. Solve for R and t and use them to reproject the points on one image to another image. 
3. Repeat this process. TODO: when to stop?

This propose-and-pick-best is called **"Random Sample Consensus", or RANSAC**. In general, RANSAC is better than least squares when there's a lot of error in the input data

## Step 3 Solve for R, and t?

E could be singular decomposed into: 
$$
E = U \Sigma V^T
$$

where E has eigen values $\sigma_1$, $\sigma_2$, $\sigma_3$. To project E onto a manifold, it will be equvalent to $diag(1,1,0)$ 

$$
\begin{align*}
\hat{t}_1 &= U R_Z\left(\frac{\pi}{2}\right) \Sigma U^\top, & R_1 &= U R_Z^\top\left(\frac{\pi}{2}\right) V^\top, \\
\hat{t}_2 &= U R_Z\left(-\frac{\pi}{2}\right) \Sigma U^\top, & R_2 &= U R_Z^\top\left(-\frac{\pi}{2}\right) V^\top.
\end{align*}
$$

Each combination of $t$ and $R$ could be a valid solution. They correspond to the below scenarios: 

<p align="center">
<img src="https://github.com/RicoJia/Omnid_Project/assets/39393023/27e5e2a9-fc12-431e-8778-77855504ee3e" height="300" width="width"/>
</p>

But only the first scenario has both canonical points' depths being positive. So, we just need to plug in R and t, and make sure that holds. This is called a "Cheirality Check" - it's done by **triangulation** [1]. Given $R$ and $t$, one can establish

$$
\begin{gather*}
Z_1 P_{c1} = Z_2 R P_{c2} + t
\\
=> 
\\
Z_1 P_{c1} \times P_{c1} = Z_2 P_{c1} \times R P_{c2} + P_{c1} \times t = 0
\end{gather*}
$$

Then, one can solve for depths $Z_1$ and $Z_2$. If they are both positive, then the solution is valid.

## [Optional] Step 4 - Homography for Co-Planar Features

If we have feature points landed on one plane, like a wall, or a floor, then, we can solve for H in $p_2 = Hp_1$, which gives $R$ and $t$. This method is a.k.a Direct Linear Transform

Recall that we have a plane


$$
\begin{gather*}
n^TP + d = 0
\\
=>
\\
-\frac{n^TP}{d}=1
\\
\end{gather*}
$$

Then plug these into $P_1$ and $P_2$

$$
\begin{gather*}
p_2 = K(RP_1+t) = K(RP_1+t(-\frac{n^T}{d})P_1)
\\
= K(R+t(-\frac{n^T}{d}))P_1
\\
= K(R+t(-\frac{n^T}{d}))K^{-1}p_1 = Hp_1
\end{gather*}
$$

Similar to the 8 point algorithm, 1 of 4 solutions could be valid. By applying the "positive depth" constraint, we can eliminate two. I'm not sure how to eliminate the last one?

## Step 5 - Do Both 8 points and Homography To Avoid De-generation

When there is no linear translation $t$, $E$ would be zero matrix too (if you multiply E and R) together. So a common practice is to calculate both homography and 8 points, then reproject them into back into the picture

### 🔎 How Do we Check Our Results?

Reprojection. Project points on 1 image point onto the other. How? 

First, we can see that the projection of one point can be mapped to multiple points, due to the scale ambiguity in translation (while rotation should be absolute). On **canonical planes**, The mapped points of $P_{c1}$ for example, are actually the line $l_2$ on the second image. Why?

<p align="center">
<img src="https://github.com/RicoJia/Omnid_Project/assets/39393023/e6f684d8-de6c-4185-af21-f878ae7d5b33" height="300" width="width"/>
</p>

Remember the epipolar constraint $P_{c2}^TEP_{c1}$ = 0? It already tells that the 3-vector $EP_{c1} = [a,b,c]$ is perpendicular to $P_{c2} = [x,y,1]$. That defines a line! $ax + by + c = 0$. So once we get this point, if the $P_2$ is close to this line, we think this projection is successful.

## Reference

[1] Nistér, D. 2004. An Efficient Solution to the Five-Point Relative Pose Problem. In *Proceedings of the 2004 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'04)*, Vol. 2. IEEE, 195–202. DOI: https://www-users.cse.umn.edu/~hspark/CSci5980/nister.pdf
