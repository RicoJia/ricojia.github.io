---
layout: post
title: Math - Quaternion
date: '2017-03-15 13:19'
subtitle: Definition of Quaternion, Quaternion Use Cases
comments: true
tags:
    - Math
---

## Definition of Quaternion

Quaternion is:

$$
\begin{gather*}
q = q_0 + q_1i + q_2 j + q_3 k
\end{gather*}
$$

i, j, k are 3 complex dimensions, where:

$$
\begin{gather*}
i^2 = j^2 = k^2 = ijk = -1
\end{gather*}
$$

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/fb78c8b7-60b7-4050-b037-8d0bf5b8cac0" height="300" alt=""/>
       </figure>
    </p>
</div>

We represent $\sigma = (i, j, k)$.

With a known rotation axis (also a unit vector) in real space, `n`, and a known roation angle $\theta$, q can be represented as

$$
\begin{gather*}
q = cos(\frac{\theta}{2}) + sin(\frac{\theta}{2}) (n \cdot \sigma)
\end{gather*}
$$

A general quaternion does NOT have to be a unit vector, however, a rotation quaternion **must be a unit vector**.

So one can see that `q` is still a unit vector from the rotation axis:

$$
\begin{gather*}
q_0 = cos(\frac{\theta}{2}), q_1 = n_x sin(\frac{\theta}{2}), q_2 = n_y sin(\frac{\theta}{2}), q_3 = n_z sin(\frac{\theta}{2})
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

### q conjugate: $\bar{q}$

$$
\begin{gather*}
\bar{q} = q_0 - q_1i - q_2 j - q_3 k
q\bar{q} = 1
\end{gather*}
$$

## Using Quaternion

If we want to rotate a vector `v` about an axis and an angle that are represented by the quaternion $q$, the rotated vector `v'` is:

$$
\begin{gather*}
v' = qv\bar{q}
\end{gather*}
$$

Where `v` is treated as a purely imaginary quaternion.

$$
\begin{gather*}
v = 0 + v_x i + v_y j + v_z k
\end{gather*}
$$

### From Quat to Axis-Angle

We always require that $\theta \in [0, \pi]$. And for cases where $\theta$ is out of $[0, \pi]$, we change the sign of q.

$$
\begin{gather*}
\theta = 2 arccos(q_0)
\\
n = [q_1, q_2, q_3]/sin(\frac{\theta}{2})
\end{gather*}
$$
