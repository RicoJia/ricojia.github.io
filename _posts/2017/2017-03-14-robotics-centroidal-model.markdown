---
layout: post
title: Robotics - The Centroidal Model
subtitle: Center of Mass, Linear Momentum, Angular Momentum of Multi-Body Systems
date: '2017-03-14 13:19'
header-img: "img/post-bg-2015.jpg"
tags:
    - Robotics
    - Math
---

## The Centroidal Model

The centroidal model describes a multi-body system (e.g., a legged robot) by its total mass, its center of mass (COM), and its total linear and angular momentum about the COM.

![](https://i.postimg.cc/T1v86nb6/download-(1).jpg)

### Center of Mass

Pick any reference frame (the origin is arbitrary). For $$n$$ bodies with masses $$m_i$$ whose own centers of mass are at $$\mathbf r_i$$, the system's center of mass is:

$$
\mathbf c = \frac{m_1 \mathbf r_1 + m_2 \mathbf r_2 + \dots + m_n \mathbf r_n}{m_1 + m_2 + \dots + m_n}
$$

Moving the origin changes the coordinates of $$\mathbf c$$, but not the physical point it refers to.

### Total Linear Momentum

$$
\mathbf p = m_1 \mathbf v_1 + m_2 \mathbf v_2 + \dots + m_n \mathbf v_n
$$

where $$\mathbf v_i$$ is the velocity of body $$i$$'s center of mass.

### Total Angular Momentum

Take two connected rigid bodies, 1 and 2, for example. Relative to the reference frame, body 1's center of mass moves with linear velocity $$\mathbf v_1$$ and the body rotates with angular velocity $$\boldsymbol\omega_1$$. Body 2 likewise has $$\mathbf v_2$$ and $$\boldsymbol\omega_2$$.

The angular momentum of body $$i$$ about the system COM $$\mathbf c$$ is:

$$
\mathbf L_i = \mathbf I_i \boldsymbol\omega_i + (\mathbf r_i - \mathbf c) \times m_i \mathbf v_i
$$

So the total angular momentum is:

$$
\boxed{\mathbf L = \sum_i \left[ \mathbf I_i \boldsymbol\omega_i + (\mathbf r_i - \mathbf c) \times m_i \mathbf v_i \right]}
$$

The first term is the body spinning about its own COM; the second is the body's COM orbiting the system COM.

### Proof

Consider a small particle $$dm$$ on body $$i$$, located at $$\boldsymbol\rho$$ relative to body $$i$$'s center of mass.

![](https://i.postimg.cc/FRB9GKsS/download-(2).jpg)

- Its position relative to the system COM is $$\mathbf r_i - \mathbf c + \boldsymbol\rho$$.
- Its velocity is $$\mathbf v_i + \boldsymbol\omega_i \times \boldsymbol\rho$$.
- By the definition of center of mass, $$\int \boldsymbol\rho \, dm = \mathbf 0$$.

The angular momentum of body $$i$$ is:

$$
\begin{aligned}
\mathbf L_i &= \int (\mathbf r_i - \mathbf c + \boldsymbol\rho) \times (\mathbf v_i + \boldsymbol\omega_i \times \boldsymbol\rho) \, dm \\
&= \int (\mathbf r_i - \mathbf c) \times \mathbf v_i \, dm
+ \int \boldsymbol\rho \times \mathbf v_i \, dm \\
&\quad + \int (\mathbf r_i - \mathbf c) \times (\boldsymbol\omega_i \times \boldsymbol\rho) \, dm
+ \int \boldsymbol\rho \times (\boldsymbol\omega_i \times \boldsymbol\rho) \, dm
\end{aligned}
$$

$$\mathbf r_i$$, $$\mathbf c$$, $$\mathbf v_i$$ and $$\boldsymbol\omega_i$$ do not depend on the particle, so they are constants in these integrals. Taking each term in turn:

1. Equivalent to a point mass at body $$i$$'s COM moving at $$\mathbf v_i$$:

$$
\int (\mathbf r_i - \mathbf c) \times \mathbf v_i \, dm = (\mathbf r_i - \mathbf c) \times m_i \mathbf v_i
$$

2. Vanishes because $$\int \boldsymbol\rho \, dm = \mathbf 0$$:

$$
\int \boldsymbol\rho \times \mathbf v_i \, dm = \left( \int \boldsymbol\rho \, dm \right) \times \mathbf v_i = \mathbf 0
$$

3. Vanishes for the same reason:

$$
\int (\mathbf r_i - \mathbf c) \times (\boldsymbol\omega_i \times \boldsymbol\rho) \, dm = (\mathbf r_i - \mathbf c) \times \left( \boldsymbol\omega_i \times \int \boldsymbol\rho \, dm \right) = \mathbf 0
$$

4. By the BAC-CAB rule, this is the body's rotational inertia about its own COM:

$$
\int \boldsymbol\rho \times (\boldsymbol\omega_i \times \boldsymbol\rho) \, dm
= \int \left( \|\boldsymbol\rho\|^2 I_3 - \boldsymbol\rho \boldsymbol\rho^T \right) dm \; \boldsymbol\omega_i
= \mathbf I_i \boldsymbol\omega_i
$$

Adding the surviving terms gives $$\mathbf L_i = \mathbf I_i \boldsymbol\omega_i + (\mathbf r_i - \mathbf c) \times m_i \mathbf v_i$$. $$\blacksquare$$

## Example: A Rod Rotating About One End

![](https://i.postimg.cc/QtHqv6s4/download-(3).jpg)

Take a uniform rod of mass $$m$$ and length $$l$$, rotating about one end with $$\boldsymbol\omega$$ perpendicular to the rod.

Inertia is independent of linear motion, so pretend the rod is only rotating. Then:

$$
\mathbf L = \int \mathbf r \times (\boldsymbol\omega \times \mathbf r) \, dm = \int \left[ \boldsymbol\omega (\mathbf r \cdot \mathbf r) - \mathbf r (\mathbf r \cdot \boldsymbol\omega) \right] dm
$$

$$\boldsymbol\omega$$ and $$\mathbf r$$ are orthogonal, so $$\mathbf r \cdot \boldsymbol\omega = 0$$:

$$
\mathbf L = \int \boldsymbol\omega \, r^2 \, dm
$$

With linear density $$\rho = m / l$$, we have $$dm = \rho \, dr$$, so:

$$
\mathbf L = \boldsymbol\omega \int_0^l \rho \, r^2 \, dr = \frac{\rho \, l^3}{3} \boldsymbol\omega
$$

Substituting $$\rho = m / l$$:

$$
\boxed{I = \frac{m l^2}{3}}
$$
