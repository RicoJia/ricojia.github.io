---
layout: post
title: Preintegration Lio
date: 2026-05-09 13:19
subtitle:
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine-Learning
---
## 1 - Time-Windowed State Estimate

Imagine we have states at four times:

$$  
x_1,\ x_2,\ x_3,\ x_4  
$$

with odometry edges

$$  
x_1-x_2,\quad x_2-x_3,\quad x_3-x_4  
$$

and an additional edge

$$  
x_1-x_3  
$$

Each state may contain position, velocity, rotation, IMU bias, etc.:

$$  
x_i = [p_i, v_i, R_i, \ldots]  
$$

In a LIO system, we usually keep only a fixed time window so that online optimization remains bounded. Suppose the window is 3 seconds. When $x_4$ arrives, the active states become

$$  
x_2,\ x_3,\ x_4  
$$

so $x_1$ must be removed. However, we cannot simply delete $x_1$. The measurements involving $x_1$ contain useful information about both $x_2$ and $x_3$. If we delete $x_1$ and all of its edges directly, that information would be lost. Instead, we marginalize $x_1$:

$$  
p(x_2,x_3,x_4) = 

\int  
p(x_1,x_2,x_3,x_4)  
,dx_1  
$$

In the linearized optimization problem, this marginalization is performed using the **Schur complement**. The result is a new **marginalized prior** on the states that remain. Because $x_1$ was connected to both $x_2$ and $x_3$, this prior generally couples $x_2$ and $x_3$. So the graph
$$  
x_1-x_2-x_3-x_4  
$$
with the additional edge
$$  
x_1-x_3  
$$
becomes

$$  
\text{marginalized prior on }x_2,x_3  
\quad+\quad  
x_2-x_3-x_4  
$$

No new information is created. The information associated with the old state $x_1$ is compressed into a prior on the states that remain in the optimization window. **The Schur complement is the mathematical operation used to perform this compression.**

---

### 1.1 - Schur Complement

For simplicity, suppose each state is a scalar:

$$  
x_1,\ x_2,\ x_3,\ x_4  
$$

Assume the measurements are

$$  
x_1 \approx 0  
$$

$$  
x_2-x_1 \approx 1  
$$

$$  
x_3-x_2 \approx 1  
$$

$$  
x_4-x_3 \approx 1  
$$

and we also have the additional measurement

$$  
x_3-x_1 \approx 2  
$$

The least-squares cost is

$$  
J =  
\frac{1}{2}  
\left[  
\frac{x_1^2}{\sigma_1^2}  
+  
\frac{(x_2-x_1-1)^2}{\sigma_{12}^2}  
+  
\frac{(x_3-x_2-1)^2}{\sigma_{23}^2}  
+  
\frac{(x_4-x_3-1)^2}{\sigma_{34}^2}  
+  
\frac{(x_3-x_1-2)^2}{\sigma_{13}^2}  
\right]  
$$

The corresponding information matrix is
 $$  
\Lambda = 

\begin{bmatrix}  
3 & -1 & -1 & 0 \\  
-1 & 2 & -1 & 0 \\  
-1 & -1 & 3 & -1 \\  
0 & 0 & -1 & 1  
\end{bmatrix}  
$$

The diagonal values represent information associated with each state. The off-diagonal values represent coupling between states. Now suppose the sliding window only keeps

$$  
x_2,\ x_3,\ x_4  
$$

and we want to eliminate $x_1$. Partition the information matrix as

$$  
\Lambda = 

\begin{bmatrix}  
\Lambda_{11} & \Lambda_{1r} \\  
\Lambda_{r1} & \Lambda_{rr}  
\end{bmatrix}  
$$

where

$$  
\Lambda_{11}=3  
$$

 $$  
\Lambda_{1r} = 

\begin{bmatrix}  
-1 & -1 & 0  
\end{bmatrix}  
$$

$$  
\Lambda_{r1} = 

\begin{bmatrix}  
-1 \\  
-1 \\  
0  
\end{bmatrix}  
$$

and

$$  
\Lambda_{rr} = 

\begin{bmatrix}  
2 & -1 & 0 \\  
-1 & 3 & -1 \\  
0 & -1 & 1  
\end{bmatrix}  
$$

The Schur complement eliminates $x_1$:

$$  
\Lambda_{\text{new}} =

\Lambda_{rr}

\Lambda_{r1}  
\Lambda_{11}^{-1}  
\Lambda_{1r}  
$$
Therefore,

$$  
\Lambda_{\text{new}} = 

\begin{bmatrix}  
\frac{5}{3} & -\frac{4}{3} & 0 \\  
-\frac{4}{3} & \frac{8}{3} & -1 \\  
0 & -1 & 1  
\end{bmatrix}  
$$

The important point is that marginalizing $x_1$ changes both the diagonal and off-diagonal terms associated with $x_2$ and $x_3$. This happens because $x_1$ was connected to both states. The information inherited from $x_1$ creates a marginalized prior on

$$  
x_2,\ x_3  
$$

rather than a prior on only $x_2$.

Another way to see this is to separate the remaining measurements from the marginalized prior (using the term prior because we haven't done graph optimization yet). Without the information from $x_1$, the remaining chain

$$  
x_2-x_3-x_4  
$$

has information matrix

$$  
\Lambda_{\text{remain}} = 

\begin{bmatrix}  
1 & -1 & 0 \\  
-1 & 2 & -1 \\  
0 & -1 & 1  
\end{bmatrix}  
$$

After marginalizing $x_1$,

$$  
\Lambda_{\text{new}} = 

\Lambda_{\text{remain}}  
+  
\Lambda_{\text{prior}}  
$$

Therefore,

$$  
\Lambda_{\text{prior}} = 

\begin{bmatrix}  
\frac{2}{3} & -\frac{1}{3} & 0 \\  
-\frac{1}{3} & \frac{2}{3} & 0 \\  
0 & 0 & 0  
\end{bmatrix}  
$$

The nonzero part of this prior is

$$  
\begin{bmatrix}  
\frac{2}{3} & -\frac{1}{3} \\  
-\frac{1}{3} & \frac{2}{3}  
\end{bmatrix}  
$$
on

$$  
x_2,\ x_3  
$$

The diagonal terms contain information about each state. The off-diagonal term

$$  
-\frac{1}{3}  
$$

means the marginalized prior also preserves a relationship between $x_2$ and $x_3$. If $x_1$ is connected only to $x_2$, marginalizing $x_1$ produces a prior mainly on $x_2$. If $x_1$ is connected to both $x_2$ and $x_3$, marginalizing $x_1$ produces a **joint prior coupling $x_2$ and $x_3$**.

In general:

> When a state is marginalized, the information carried through that state is transferred to its remaining neighboring states.

So

$$  
x_1  
\rightarrow  
x_2,\ x_3  
$$

becomes

$$  
\text{marginalized prior on }x_2,x_3  
$$

This is the main purpose of the Schur complement in sliding-window optimization: **remove an old state while preserving its effect on the states that remain.**