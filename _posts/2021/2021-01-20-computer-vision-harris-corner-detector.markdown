---
layout: post
title: Computer Vision - Harris Corner Detector
date: '2021-01-20 13:19'
subtitle: Do Not Cut Corners on Corner Detector - It Will See It
comments: true
tags:
    - Computer Vision
---

## Basic Idea

When a specified window truly has a corner, if we shift a window (region of interest) along any direction (+/- x or y direction) slightly by small displacements, $u$, $v$, we could expect a noticeable difference in the **sum of squared difference** in intensity $I$ **before and after** the shift:

$$
\begin{gather*}
E(u,v) = \sum_{X,Y} (I(x+u, y+v) - I(x,y))^2
\end{gather*}
$$


<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/4a7bb331-796d-4d15-8659-94093b390360" height="300" alt=""/>
    </figure>
</p>
</div>

In real life, an unweighted window doesn't work too well mostly because center points may deserve more focus than farther points, otherwise we might accumulate too much noise. So, we can apply a window function to emphasize that. People like to use a Gaussian window $w(x,y)$

$$
\begin{gather*}
E(u,v) = \sum_{X,Y} w(x,y)(I(x+u, y+v) - I(x,y))^2
\end{gather*}
$$

In the day and age when compute was limited, to make the above faster, we can make use of image gradients along $x$ and $y$: $I_x$, $I_y$. That's because image gradients along each direction can be calculated in 1 pass. To do that, let's meet our old friend, **Taylor Expansion**:

$$
\begin{gather*}
I(x+u, y+v) = I(x,y) + I_x(x, y)u + I_y(x,y)v
\end{gather*}
$$

That yields:

$$
\begin{gather*}
(I(x+u, y+v) - I(x,y))^2 = (I_x(x, y)u + I_y(x,y)v)^2
\\
= 
\\
\begin{bmatrix}
u & v
\end{bmatrix}

\begin{bmatrix}
I_x^2(x,y) & I_x(x,y)I_y(x,y) \\
I_x(x,y)I_y(x,y) & I_y^2(x,y)
\end{bmatrix}

\begin{bmatrix}
u \\
v
\end{bmatrix}
\end{gather*}
$$

Usually, we write use the **second moment matrix** to represent the gradient matrix:

$$
\begin{gather*}
M = \begin{bmatrix}
I_x^2(x,y) & I_x(x,y)I_y(x,y) \\
I_x(x,y)I_y(x,y) & I_y^2(x,y)
\end{bmatrix}
\end{gather*}
$$

Counting the smoothing window in, the total **summed of squared difference** is

$$
\begin{gather*}
E(u,v) = \sum_{X,Y} w(x,y)

\begin{bmatrix}
u & v
\end{bmatrix}

M

\begin{bmatrix}
u \\
v
\end{bmatrix}
\end{gather*}
$$

## What Second Moment Matrix Means

An ellipsoid centered at origin is generally:

$$
\begin{gather*}
ax^2 + 2bxy + cy^2 = 0
\end{gather*}
$$

This ellipsoid can be written as:

$$
\begin{gather*}
\begin{bmatrix}
x & y
\end{bmatrix}

\begin{bmatrix}
a & b \\
b & c
\end{bmatrix}

\begin{bmatrix}
x \\
y
\end{bmatrix}

\end{gather*}
$$

**So, the above sum of squared difference really represents a WEIGHTED ELLIPSOID!!** Visualizing in 3D, the ellipsoid looks like:

<p align="center">
<img src="https://user-images.githubusercontent.com/39393023/131152407-d01721e2-d83f-4546-983f-f70eb5965b8f.png" height="200" width="width"/>
</p>

Mathematically, one principle direction of the ellipsoid is represented by the **eigen vector** of the matrix $M$. The length along that principle direction is the corresponding eigen value of $M$. So when we move the window in:

- A flat region: no high sum of squared difference is expected. **Neither eigen value should be large**
- An edge: we expect high sum of squared difference along the direction perpendicular to the edge. So 1 eigen value of M should be large, the other is small
- A corner: we would expect high sum of squared difference along both $x$ and $y$ directions. So both eigen values should be high

## How To Measure Harris Corner Response

Carrying the spirit of ellipsoid and eigen values, the "Harris Corner Response", or R, measures if both eigen values are large: 

$$
\begin{gather*}
R = det(M) - \alpha \cdot trace(M)^2 = \lambda_1 \lambda_2 - \alpha (\lambda_1 + \lambda_2)^2
\end{gather*}
$$

- Recall: $det(M) = \lambda_1 \lambda_2$
- $\alpha \in [0.04, 0.06]$ (to balances the sensitivity between **edges vs corners**
- $R \approx 0$ for flat region, $R < 0$ for an edge (one large lambda one small lambda), and $R > 0$ for a corner

So the general workflow is:

1. Calculate image gradient using the Sobel Operator
2. For each pixel, compute its second moment matrix, $M$, within a small window
3. Compute $R$ for each pixel, then threshold it, and find local maxima with Non Maximum Suppresion to ensure the most prominent corners are selected.


### Properties Of Harris Corner

- Invariant to rotation: 

<p align="center">
<img src="https://user-images.githubusercontent.com/39393023/131229240-8aea8978-70c6-438b-b360-46c095605c87.JPEG" height="200" width="width"/>
</p>

- Not invariant to scale changes

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/b8334fc4-3565-4e5e-a125-cc569eb5eacb" height="200" alt=""/>
    </figure>
</p>
</div>

### OpenCV Implementation

CV2 Harris Corner uses this function ```R=det(M)−k(trace(M))2```, when R<0, one lambda is much larger than the other. when R>0 and R is large, that's a corner.

- ksize means the size of sobel operator kernel

## Final Remarks

Shi-Tomasi corner (1994) basically uses the same M matrix. The difference is its cornerness is to find $min(\lambda_1, \lambda_2)$, then find local maximums. ?? This is called **Good Features to Track**