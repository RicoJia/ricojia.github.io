---
layout: post
title: Math - Zero Order Hold And Discretization
subtitle: Continuous to Discrete State Space, Matrix Exponential, Euler vs. ZOH
date: '2017-03-18 13:19'
header-img: "img/post-bg-2015.jpg"
tags:
    - Math
    - Robotics
---

## Zero Order Hold

When converting a continuous-time state space transition model to its discrete equivalent, we need to integrate the state transition over the sampling interval $$\Delta t$$.

- One intuitive way to do it is **Euler integration**, which assumes all state derivatives are constant over the interval.
- A better way to do it is **Zero Order Hold (ZOH)**, which assumes only the control input is constant over the interval, and **integrates the state variables** accordingly.

Note, ZOH itself is an assumption, not an integration method. Under ZOH, one can still do RK4 or Euler integration. RK4 approximates the integral to the fourth order.

## Simple Motivating Example - Free Flight With Drag

One example is a drone. Its acceleration is actually impacted by drag: the higher the velocity, the higher the drag. To begin with, assume there is zero control input, so the drone is free flying:

$$
v' = a v
$$

where $$a < 0$$, since drag opposes the motion. Then

$$
\frac{dv}{dt} = a v \quad \Rightarrow \quad \frac{1}{v} dv = a \, dt
$$

Integrate over the sampling interval, from $$v_0$$ at $$t = 0$$ to $$v$$ at $$t = \Delta t$$:

$$
\int_{v_0}^{v} \frac{1}{v} dv = \int_0^{\Delta t} a \, dt
\quad \Rightarrow \quad
\ln \frac{v}{v_0} = a \Delta t
$$

So

$$
\boxed{v = e^{a \Delta t} v_0}
$$

In comparison, Euler integration simply does:

$$
v = v_0 + v' \Delta t = v_0 + a v_0 \Delta t = (1 + a \Delta t) v_0
$$

Actually, Taylor expand ZOH:

$$
v = \left(1 + a \Delta t + \frac{(a \Delta t)^2}{2!} + \dots \right) v_0
$$

As we can see, **Euler integration is the first-order approximation of ZOH**.

## Example - Velocity Model With Control Input

Now add a control input $$u$$, which is held constant at $$u_k$$ over the interval (that's the ZOH assumption):

$$
v' = a v + b u
$$

So

$$
\frac{dv}{dt} - a v = b u
$$

Multiply both sides by $$e^{-at}$$:

$$
e^{-at} \frac{dv}{dt} - e^{-at} a v = e^{-at} b u
$$

Note that the left side is exactly the derivative of a product (product rule):

$$
\frac{d}{dt}\left[ e^{-at} v \right] = e^{-at} b u
$$

Integrate both sides over $$[0, \Delta t]$$. Remember, $$v$$ is actually also a function of $$t$$, $$v(t)$$, with $$v(0) = v_k$$ and $$v(\Delta t) = v_{k+1}$$:

$$
\int_0^{\Delta t} \frac{d}{dt}\left[ e^{-at} v \right] dt = \int_0^{\Delta t} e^{-at} b u_k \, dt
$$

So

$$
e^{-a \Delta t} v_{k+1} - v_k = b u_k \, \frac{1 - e^{-a \Delta t}}{a}
$$

Solving for $$v_{k+1}$$:

$$
\boxed{v_{k+1} = e^{a \Delta t} v_k + b u_k \, \frac{e^{a \Delta t} - 1}{a}}
$$

## State Space Representation

In a state space model, the state $$\mathbf x$$ and the control $$\mathbf u$$ are vectors, and $$A$$, $$B$$ are matrices. Again, $$\mathbf u$$ is held constant at $$\mathbf u_k$$ over the interval:

$$
\mathbf x' = A \mathbf x + B \mathbf u
$$

So

$$
\frac{d\mathbf x}{dt} - A \mathbf x = B \mathbf u
$$

Multiply both sides by the matrix exponential $$e^{-At}$$ (on the left, since matrices don't commute):

$$
e^{-At} \frac{d\mathbf x}{dt} - e^{-At} A \mathbf x = e^{-At} B \mathbf u
$$

So similarly, the left side is the derivative of a product:

$$
\frac{d}{dt}\left[ e^{-At} \mathbf x \right] = e^{-At} B \mathbf u
$$

Evaluating over $$[0, \Delta t]$$, with $$\mathbf x(0) = \mathbf x_k$$ and $$\mathbf x(\Delta t) = \mathbf x_{k+1}$$:

$$
e^{-A \Delta t} \mathbf x_{k+1} - \mathbf x_k = \int_0^{\Delta t} e^{-A\tau} B \mathbf u_k \, d\tau
$$

Solving for $$\mathbf x_{k+1}$$:

$$
\mathbf x_{k+1} = e^{A \Delta t} \mathbf x_k + e^{A \Delta t} \int_0^{\Delta t} e^{-A\tau} \, d\tau \, B \mathbf u_k
$$

Since $$e^{A \Delta t}$$ is independent from $$\tau$$, we can pull $$e^{A \Delta t}$$ inside the integral. Then substituting $$\sigma = \Delta t - \tau$$ gives the standard discrete form:

$$
\boxed{\mathbf x_{k+1} = A_d \mathbf x_k + B_d \mathbf u_k, \quad A_d = e^{A \Delta t}, \quad B_d = \int_0^{\Delta t} e^{A\sigma} \, d\sigma \, B}
$$

If $$A$$ is invertible, $$B_d = A^{-1}\left(e^{A \Delta t} - I\right) B$$, which matches the scalar result $$b \, \frac{e^{a \Delta t} - 1}{a}$$ above.

### Singular A

But $$A$$ can be singular (e.g., a pure integrator like position from velocity), so $$A^{-1}$$ doesn't exist. Instead, use the series. The matrix exponential is simply

$$
e^{A \Delta t} = I + A \Delta t + \frac{(A \Delta t)^2}{2!} + \dots
$$

So, integrating term by term:

$$
\begin{aligned}
B_d &= \int_0^{\Delta t} \left( I + A\sigma + \frac{(A\sigma)^2}{2!} + \dots \right) d\sigma \, B \\
&= \left( I \Delta t + \frac{A \Delta t^2}{2!} + \frac{A^2 \Delta t^3}{3!} + \dots \right) B
\end{aligned}
$$

No inverse needed, so this works for any $$A$$.

Therefore, combining $$A_d \mathbf x_k + B_d \mathbf u_k$$ and grouping by powers of $$\Delta t$$, the final state space representation is:

$$
\begin{aligned}
\mathbf x_{k+1} = \mathbf x_k &+ (A \mathbf x_k + B \mathbf u_k) \Delta t \\
&+ (A^2 \mathbf x_k + A B \mathbf u_k) \frac{\Delta t^2}{2!} \\
&+ (A^3 \mathbf x_k + A^2 B \mathbf u_k) \frac{\Delta t^3}{3!} + \dots
\end{aligned}
$$

Note, truncating after the $$\Delta t$$ term gives $$\mathbf x_{k+1} = \mathbf x_k + (A \mathbf x_k + B \mathbf u_k) \Delta t$$, which is exactly Euler integration. Same conclusion as the scalar case: **Euler integration is the first-order approximation of ZOH**.
