---
layout: post
title: Math - Trace And Frobenius Norm Command
date: '2017-02-28 13:19'
subtitle: 
comments: true
tags:
    - Math
---

## Trace Properties

1. Proving $\mathrm{tr}(AB) = \mathrm{tr}(BA)$ (cyclic shifting)

$$
\begin{align*}
(AB)_{ii} &= \sum_j A_{ij} B_{ji}, \\
\mathrm{tr}(AB) &= \sum_i \sum_j A_{ij} B_{ji}, \\
\mathrm{tr}(BA) &= \sum_j \sum_i B_{ji} A_{ij}.
\end{align*}
$$


2. Proving $\|A\|_F^2 = \mathrm{tr}(A^\top A)$

$$
\begin{align*}
[A^\top A]_{ii} &= \sum_j A_{ji} A_{ji}, \\
\mathrm{tr}(A^\top A) &= \sum_i \sum_j A_{ji} A_{ji} = \|A\|_F^2
\end{align*}
$$


## Frobenius Norm

Frobenius norm = $\sum_i \sum_j (a_{ij} * a_{ij})$

E.g., a common task in lidar is if we have an estimate $R$ of an SO(3) matrix, we want to find the closest SO(3) matrix $X$ with the lowest Frobenius norm. That is:

Proving $\|X-R\|_F^2 = \|X\|_F^2 + \|R\|_F^2 - 2\mathrm{tr}(X^\top R)$ and deriving $\arg\max(\mathrm{tr}(X^\top R))$

$$
\begin{align*}
\|X-R\|_F^2 &= \mathrm{tr}((X-R)^\top(X-R)) \\
&= \mathrm{tr}(X^\top X - X^\top R - R^\top X + R^\top R) \\
&= \mathrm{tr}(X^\top X) - \mathrm{tr}(X^\top R) - \mathrm{tr}(R^\top X) + \mathrm{tr}(R^\top R) \\
&= \|X\|_F^2 - \mathrm{tr}(X^\top R) - \mathrm{tr}(X^\top R) + \|R\|_F^2 \\
&= \|X\|_F^2 + \|R\|_F^2 - 2\mathrm{tr}(X^\top R)
\end{align*}
$$

To minimize $\|X-R\|_F^2$, we need to maximize $\mathrm{tr}(X^\top R)$ since $\|X\|_F^2$ and $\|R\|_F^2$ are constants. Therefore:

$$\arg\min_R \|X-R\|_F^2 = \arg\max_R \mathrm{tr}(X^\top R)$$

Now, we can perform SVD on $R$:

$$R = U \Sigma V^\top$$

To find $X$, we define an intermediate variable:

$$Y = U^\top X V$$

Since $U$ and $V$ are orthonormal matrices, they are in $\text{SO}(3)$. Then, $Y$ is also in $\text{SO}(3)$.

So now $\mathrm{tr}(X^\top R)$ becomes:

$$
\begin{align*}
\mathrm{tr}(X^\top R) &= \mathrm{tr}((UYV^\top)^\top (U \Sigma V^\top)) \\
&= \mathrm{tr}(VY^\top U^\top U \Sigma V^\top) \\
&= \mathrm{tr}(VY^\top \Sigma V^\top)
\end{align*}
$$

Because of the cyclic shifting property (see above):

$$
\begin{align*}
\mathrm{tr}(VY^\top \Sigma V^\top) &= \mathrm{tr}(Y^\top \Sigma V^\top V) \\
&= \mathrm{tr}(Y^\top \Sigma)
\end{align*}
$$

