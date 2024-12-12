---
layout: post
title: Math - Quaternion
date: '2017-03-13 13:19'
subtitle: Definition of Quaternion, Quaternion Use Cases
comments: true
tags:
    - Math
---

## Definition of Quaternion

Quaternions can be represented as:

$$
\begin{gather*}

q = [s, x, y, z] = s + xi + yj + zk = s + v
v = (i, j, k)
\end{gather*}
$$

i, j, k are 3 complex dimensions, where:

$$
\begin{gather*}
i^2 = j^2 = k^2 = ijk = -1
\end{gather*}
$$

### i, j, k Are Not Commutative

$$
\begin{gather*}
ij = -ij(kk) = (-ijk)k = k  \\
jk = -(ii)jk = -i(ijk) = i  \\
ki = -ki*(jj) = -k*k*j = j   \\
\end{gather*}
$$

Note i, j, k are **NOT commutative.**

$$
\begin{gather*}
ji = j * jk = -k    \\
kj = k * ki = -i    \\
ik = i * ij = -j
\end{gather*}
$$

### Product of Quaternions

The product of two quaternions are:

$$
\begin{gather*}
q_a q_b = s_a s_b - x_a x_b - y_a y_b - z_a z_b \\
+ (s_a x_b + x_a s_b + y_a z_b - z_a y_b) \, i \\
+ (s_a y_b - x_a z_b + y_a s_b + z_a x_b) \, j \\
+ (s_a z_b + x_a y_b - y_a x_b + z_a s_b) \, k.
\end{gather*}
$$

This can be also written in the form of vectors `s` and `v`:

$$
\begin{gather*}
q_a q_b = [s_a s_b - v_a^T v_b, s_a v_b + s_b v_a + v_a \times v_b]
\end{gather*}
$$

Note that the $v_a \times v_b$ is not commutative, so the quaternion is not commutative, unless $v_a, v_b$ are co-linear

### Length of q

$$
\begin{gather*}
|q| = \sqrt{(s^2 + x^2 + y^2 + z^2)} = \sqrt{s^2 + v^T v}
\end{gather*}
$$

One can prove that the length of a product is the product of the lengths

$$
\begin{gather*}
|q_a q_b| = |q_a| |q_b|
\end{gather*}
$$

### q conjugate: $q^{*}$

$$
\begin{gather*}
q^{*} = s - xi - yj - zk = [s, -v]
\\
q^{*}q = [s_a s_b + v^T v, 0] = q q^{*}
\end{gather*}
$$

Because $q^{*}q = [s_a s_b + v^T v, 0]$, we can see that the inverse of q is

$$
\begin{gather*}
q^{-1} = \frac{q}{|q|}
\\
q q^{-1} = \frac{[s_a s_b + v^T v, 0]}{|q|^2}  = 1
\end{gather*}
$$

#### Products' Conjugate is the Product of Conjugates

$$
\begin{gather*}
(q_a q_b)^{-1} = \frac{q_a q_b}{|q_a q_b|^2} = \frac{q_a}{|q_a|^2} \frac{q_b}{|q_b|^2} = q_a^{-1} q_b^{-1}
\end{gather*}
$$

--------------------------------------------------

## Using Quaternion

### Rotations With Quaternion

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/fb78c8b7-60b7-4050-b037-8d0bf5b8cac0" height="300" alt=""/>
       </figure>
    </p>
</div>

For a point `(x, y, z)`, `p' = Rp`. $p' = qpq^{-1}$, where `p'` has zero real part. Here is why:

- Let's define 2 matrices: (the only difference is the sign of $v^{\land}$)

$$
\begin{gather*}
q^{+} = \begin{bmatrix}
s & -v^T \\
v & sI + v^{\land}
\end{bmatrix}
\end{gather*}
$$

$$
\begin{gather*}
q^{\oplus} = \begin{bmatrix}
s & -v^T \\
v & sI - v^{\land}
\end{bmatrix}
\end{gather*}
$$

We can prove that $q_1^{+} q_2 = q_1 q_2$

$$
\begin{gather*}
\begin{bmatrix}
s_1 & -\mathbf{v}_1^\top \\
\mathbf{v}_1 & s_1 \mathbf{I} + \mathbf{v}_1^\wedge
\end{bmatrix}
\begin{bmatrix}
s_2 \\
\mathbf{v}_2
\end{bmatrix}
=
\begin{bmatrix}
-\mathbf{v}_1^\top \mathbf{v}_2 + s_1 s_2 \\
s_1 \mathbf{v}_2 + s_2 \mathbf{v}_1 + \mathbf{v}_1^\wedge \mathbf{v}_2
\end{bmatrix}
= q_1 q_2
\end{gather*}
$$

Similarly, $q_1^{+} q_2 = q_1 q_2 = q_2^{\oplus} q_1$. So

$$
\begin{gather*}
p' = qpq^{-1} = q^{+} p^{+} q^{-1} = q^{+} q^{-1 \oplus} p
\end{gather*}
$$

Then, we can write out $q^{+} q^{-1 \oplus}$, using that **a rotation quaternion is unit length**

$$
\begin{gather*}
q^+ \left( q^{-1} \right)^\oplus =
\begin{bmatrix}
s & -\mathbf{v}^\top \\
\mathbf{v} & s \mathbf{I} + \mathbf{v}^\wedge
\end{bmatrix}
\begin{bmatrix}
s & \mathbf{v}^\top \\
-\mathbf{v} & s \mathbf{I} + \mathbf{v}^\wedge
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 \\
0^\top & \mathbf{v}\mathbf{v}^\top + s^2 \mathbf{I} + 2s\mathbf{v}^\wedge + (\mathbf{v}^\wedge)^2
\end{bmatrix}
\end{gather*}
$$

Here, it's easy to see that $p' = qpq^{-1} = q^{+} q^{-1 \oplus} p$ has a zero real part.

#### Relationship with Rotation Matrix

Rotation Matrix can be represented as the last element in $q^{+} q^{-1 \oplus}$

$$
\begin{gather*}
p' = qpq^{-1} = v v^{T} + s^2 I + 2sv^{\land} + (v^{\land})^2 p = Rp
\end{gather*}
$$

**So, we also have $R = v v^{T} + s^2 I + 2sv^{\land} + (v^{\land})^2$.**

So we can represent trace of `R` as the real part of quaternion `q`, `s`

$$
\begin{gather*}
tr(R) = tr (v v^{T} + s^2 I + 2sv^{\land}) + tr((v^{\land})^2)
\\
\text{since:}
\\ v^{\land} = |v|a => (v^{\land})^2 = |v|^2 (a a^T - I)

\\
\text{we have:}
\\
tr((v^{\land})^2) =  |v|^2 (1 - 3) = -2 |v|^2

\\
\text{so}

\\ tr(R)
\\= (v_1^2 + v_2^2 + v_3^2) + 3s^2 + 0 - 2(v_1^2 + v_2^2 + v_3^2)
\\ = (1-s^2) + 3s^2 - 2(1-s^2)
\\ = 4s^2 - 1
\end{gather*}
$$

#### From Quaternion To The Rotation Angle, $\theta$

Since we know that:

$$
\begin{gather*}
\theta = arccos(\frac{tr(R) - 1}{2}) = arccos(2s^2 - 1)
\end{gather*}
$$

We always require that $\theta \in [0, \pi]$. And for cases where $\theta$ is out of $[0, \pi]$, we change the sign of q. So we can know the rotation angle, $\theta$:

$$
\begin{gather*}
cos\theta = 2s^2 - 1 = 2 cos \frac{\theta}{2} ^2 - 1
\end{gather*}
$$

So:

$$
\begin{gather*}
s = cos \frac{\theta}{2}
\\ =>
\\ \theta = 2arccos(s)
\end{gather*}
$$

#### From Quaternion To The Rotation Axis

**The virtual part of the rotation quaternion q, v** is actually on the rotation axis. Here is the proof:

$$
\begin{gather*}
R v = q^{+} q^{-1 \oplus}[0, v] = v
\end{gather*}
$$

Only a point on the rotation axis itself will not move by rotation. So the rotation axis (unit vector) is simply:

$$
\begin{gather*}
[n_x, n_y, n_y] = \frac{v}{|v|}
\\ = \frac{v}{\sqrt{1 - s^2}}
\\ = \frac{v}{\sqrt{1 - s^2}}
\\ =>
\\ [n_x, n_y, n_y] = \frac{v}{sin(\frac{\theta}{2})}

\end{gather*}
$$

Most libraries will use quaternion for rotations as it only requires 4 numbers. However, as a user, we don't need to worry too much about the lower level implementation details

## TODO

With a known rotation axis (also a unit vector) in real space, `n`, and a known roation angle $\theta$, q can be represented as

$$
\begin{gather*}
q = cos(\frac{\theta}{2}) + sin(\frac{\theta}{2}) (n \cdot v)
\end{gather*}
$$

A general quaternion does NOT have to be a unit vector, however, a rotation quaternion **must be a unit vector**.

So one can see that `q` is still a unit vector from the rotation axis:

$$
\begin{gather*}
q_0 = cos(\frac{\theta}{2}), q_1 = n_x sin(\frac{\theta}{2}), q_2 = n_y sin(\frac{\theta}{2}), q_3 = n_z sin(\frac{\theta}{2})
\end{gather*}
$$

## Kinematics Using Quaternions

Since we have $qq^{*} = 1$, we can get the derivative of:

$$
\begin{gather*}
q'q^{*} + qq'^{*} = 0
\\ =>
\\ q'q^{*} = -qq'^{*} = -(q'q^{*})^{*}
\end{gather*}
$$

So we can see that `q'q^{*}` must be a pure imaginary number $q'q^{*} = \bar{w} = [0, w_1, w_2, w_3]$
