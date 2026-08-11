---
layout: post
title: Math - Sherman-Morrison-Woodbury Equation
date: 2017-03-05 13:19
subtitle: How Kalman Filter Reduces Computation For Inverse
comments: true
tags:
  - Math
---
The Sherman-Morrison-Woodbury Equation is:

$$  
\boxed{  
AB(D+CAB)^{-1} = 

\left(A^{-1}+BD^{-1}C\right)^{-1}BD^{-1}  
}  
$$

Assume all displayed inverses exist.

## Direct proof

Define

$$  
X=AB(D+CAB)^{-1}.  
$$

Multiply by the matrix on the left:

$$  
\begin{aligned}  
\left(A^{-1}+BD^{-1}C\right)X  
&=  
\left(A^{-1}+BD^{-1}C\right)  
AB(D+CAB)^{-1}  
\\  
&=  
\left(B+BD^{-1}CAB\right)  
(D+CAB)^{-1}  
\\  
&=  
BD^{-1}(D+CAB)(D+CAB)^{-1}  
\\  
&=  
BD^{-1}.  
\end{aligned}  
$$

Therefore,

$$  
X = 

\left(A^{-1}+BD^{-1}C\right)^{-1}  
BD^{-1}.  
$$

Since

$$  
X=AB(D+CAB)^{-1},  
$$

we obtain

$$  
\boxed{  
AB(D+CAB)^{-1}

\left(A^{-1}+BD^{-1}C\right)^{-1}BD^{-1}  
}.  
$$
## Connection to the Kalman filter

Substitute

$$  
A=P,  
\qquad  
B=H^\top,  
\qquad  
C=H,  
\qquad  
D=R.  
$$

Note, H is `Nx18`, where is the number of constraints (e.g., ICP / NDT observations ). P is `18 x 18`

Then

$$  
\boxed{  
\left(P^{-1}+H^\top R^{-1}H\right)^{-1}  
H^\top R^{-1} = 

PH^\top(HPH^\top+R)^{-1}  
}.  
$$

The right-hand side is the Kalman gain:

$$  
\boxed{  
K=PH^\top(HPH^\top+R)^{-1}  
}.  
$$
Computing the inverse of `NxN` is hard!

Starting from the least-squares problem,

$$  
J(x) = 

\frac{1}{2}(x-\bar{x})^\top P^{-1}(x-\bar{x})  
+  
\frac{1}{2}(z-Hx)^\top R^{-1}(z-Hx),  
$$

define

$$  
\delta x=x-\bar{x},  
\qquad  
r=z-H\bar{x}.  
$$

Setting the gradient to zero gives

$$  
\left(P^{-1}+H^\top R^{-1}H\right)\delta x = 

H^\top R^{-1}r.  
$$

Therefore,

$$  
\delta x =

\left(P^{-1}+H^\top R^{-1}H\right)^{-1}  
H^\top R^{-1}r.  
$$
Computing the inverse of`18x18` and is easier!