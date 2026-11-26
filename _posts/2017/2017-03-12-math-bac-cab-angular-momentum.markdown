---
layout: post
title: Math - BAC-CAB Rule And Angular Momentum
subtitle: Cross Product, Vector Triple Product, Angular Velocity, Inertia Tensor
date: '2017-03-12 13:19'
header-img: "img/post-bg-2015.jpg"
tags:
    - Math
---

## Cross Product

For two 3D vectors, the cross product is:

$$
\mathbf a \times \mathbf b =
\begin{bmatrix} a_yb_z-a_zb_y\\ a_zb_x-a_xb_z\\ a_xb_y-a_yb_x \end{bmatrix}
$$

A handy way to remember it is the determinant form:

$$
\mathbf a \times \mathbf b =
\begin{vmatrix}
\mathbf i & \mathbf j & \mathbf k \\
a_x & a_y & a_z \\
b_x & b_y & b_z
\end{vmatrix}
$$

### Geometric Meaning

- The result is perpendicular to both $$\mathbf a$$ and $$\mathbf b$$.
- Its magnitude is the area of the parallelogram spanned by $$\mathbf a$$ and $$\mathbf b$$:

$$
\|\mathbf a \times \mathbf b\| = \|\mathbf a\| \|\mathbf b\| \sin\theta
$$

- Its direction follows the right-hand rule:

![](https://i.postimg.cc/tT0cRspb/download.jpg)

E.g., $$\mathbf z = \mathbf x \times \mathbf y$$:

![](https://i.postimg.cc/CLKcgGkH/g-X4Cbu-HCk0e-Teg-GV9Md-Ao-Mqz-DIX8scij-FM579rv-Ztr-IIxdqi-Wtzf-ETDQ59wy-Wzq-Pl-G48a9e-KE4wjceq-R6f7.jpg)

### Properties

- Anti-commutative: $$\mathbf a\times\mathbf b= -\mathbf b\times\mathbf a$$
- A vector crossed with itself (or any parallel vector) is zero: $$\mathbf a \times \mathbf a = \mathbf 0$$
- Distributive: $$\mathbf a \times (\mathbf b + \mathbf c) = \mathbf a \times \mathbf b + \mathbf a \times \mathbf c$$
- **Not** associative: $$\mathbf a \times (\mathbf b \times \mathbf c) \neq (\mathbf a \times \mathbf b) \times \mathbf c$$ in general. The BAC-CAB rule below tells us what the triple product actually is.

### Skew-Symmetric Matrix Form

The cross product with $$\mathbf a$$ is a linear map, so it can be written as a matrix multiplication:

$$
\mathbf a \times \mathbf b = [\mathbf a]_\times \mathbf b, \quad
[\mathbf a]_\times =
\begin{bmatrix}
0 & -a_z & a_y \\
a_z & 0 & -a_x \\
-a_y & a_x & 0
\end{bmatrix}
$$

$$[\mathbf a]_\times$$ is skew-symmetric: $$[\mathbf a]_\times^T = -[\mathbf a]_\times$$. This form shows up a lot in rotations and Lie groups.

## BAC-CAB Rule

The vector triple product can be expanded as:

$$
\mathbf a \times (\mathbf b \times \mathbf c) = \mathbf b (\mathbf a \cdot \mathbf c) - \mathbf c (\mathbf a \cdot \mathbf b)
$$

Hence the name "BAC - CAB".

### Proof

Let

$$
\mathbf a= \begin{bmatrix}a_x\\a_y\\a_z\end{bmatrix}, \quad
\mathbf b= \begin{bmatrix}b_x\\b_y\\b_z\end{bmatrix}, \quad
\mathbf c= \begin{bmatrix}c_x\\c_y\\c_z\end{bmatrix}
$$

Then

$$
\mathbf b\times\mathbf c= \begin{bmatrix} b_yc_z-b_zc_y\\ b_zc_x-b_xc_z\\ b_xc_y-b_yc_x \end{bmatrix}
$$

Write out the $$x$$ component of $$\mathbf a \times (\mathbf b \times \mathbf c)$$:

$$
\begin{aligned}
[\mathbf a\times(\mathbf b\times\mathbf c)]_x &= a_y(b_xc_y-b_yc_x) - a_z(b_zc_x-b_xc_z)
\end{aligned}
$$

Expand and regroup:

$$
= b_x(a_yc_y+a_zc_z) - c_x(a_yb_y+a_zb_z)
$$

Now, add and subtract $$a_xb_xc_x$$:

$$
= b_x(a_xc_x+a_yc_y+a_zc_z) - c_x(a_xb_x+a_yb_y+a_zb_z)
$$

The terms inside the parentheses are simply dot products:

$$
= b_x(\mathbf a\cdot\mathbf c) - c_x(\mathbf a\cdot\mathbf b)
$$

Applying the same procedure to the $$y$$ and $$z$$ components gives:

$$
\boxed{ \mathbf a\times(\mathbf b\times\mathbf c) = \mathbf b(\mathbf a\cdot\mathbf c) - \mathbf c(\mathbf a\cdot\mathbf b) }
$$

## Angular Velocity

- Linear momentum: $$\mathbf p = m \mathbf v$$
- Angular velocity $$\boldsymbol\omega$$ relates to the linear velocity of a point at position $$\mathbf r$$ by:

$$
\boxed{\mathbf v=\boldsymbol\omega\times\mathbf r}
$$

So for a rigid body translating with $$\mathbf v_0$$ while rotating, the instantaneous velocity of a point is:

$$
\mathbf v = \mathbf v_0 + \boldsymbol\omega \times \mathbf r
$$

## Angular Momentum

Angular momentum is defined as:

$$
\mathbf L = \mathbf r \times \mathbf p, \quad \mathbf p = m \mathbf v
$$

Plug in $$\mathbf v = \boldsymbol\omega \times \mathbf r$$ and apply the BAC-CAB rule:

$$
\begin{aligned}
\mathbf L &= m\, \mathbf r \times (\boldsymbol\omega \times \mathbf r) \\
&= m\, \boldsymbol\omega (\mathbf r \cdot \mathbf r) - m\, \mathbf r (\mathbf r \cdot \boldsymbol\omega) \\
&= m \|\mathbf r\|^2 \boldsymbol\omega - m\, \mathbf r \mathbf r^T \boldsymbol\omega \\
&= m \left( \|\mathbf r\|^2 I - \mathbf r \mathbf r^T \right) \boldsymbol\omega
\end{aligned}
$$

The trick is rewriting $$\mathbf r (\mathbf r \cdot \boldsymbol\omega)$$ as the matrix product $$\mathbf r \mathbf r^T \boldsymbol\omega$$, which lets us factor out $$\boldsymbol\omega$$. So:

$$
\mathbf L = \mathbf I \boldsymbol\omega
$$

where $$\mathbf I$$ is the **inertia tensor**:

$$
\mathbf I = m
\begin{bmatrix}
y^2 + z^2 & -xy & -xz \\
-xy & x^2 + z^2 & -yz \\
-xz & -yz & x^2 + y^2
\end{bmatrix}
$$

The inertia tensor can be thought of as the "rotating mass": just as $$\mathbf p = m \mathbf v$$ for linear motion, $$\mathbf L = \mathbf I \boldsymbol\omega$$ for rotational motion.

### Conservation of Angular Momentum

![](https://nuclear-power.com/wp-content/uploads/2016/11/conservation-angular-momentum-min.png)
