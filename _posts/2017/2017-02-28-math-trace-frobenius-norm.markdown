---
layout: post
title: Math - Trace, Determinant, Frobenius Norm
date: '2017-02-28 13:19'
subtitle: Von-Neumann's Trace Inequality
comments: true
tags:
    - Math
---

## Determinant

- $\det(AB) = \det(A)\,\det(B)$
- For an orthogonal matrix $Q$ (i.e., $Q^\top Q = I$), we have $\det(Q) = \pm 1$:

$$
\begin{align*}
& Q^\top Q &= I \\
& \Rightarrow\; \det(Q^\top Q) &= \det(I) = 1 \\
& \Rightarrow\; \det(Q^\top)\,\det(Q) &= 1 \\
& \Rightarrow\; \det(Q)\,\det(Q) &= 1 \quad (\text{since } \det(Q^\top) = \det(Q)) \\
& \Rightarrow\; (\det Q)^2 &= 1 \\
& \Rightarrow\; \det Q &= \pm 1.
\end{align*}
$$

In particular, if $Q \in SO(n)$ (special orthogonal group), then $\det Q = 1$.

## Trace Properties

1. Proving $\mathrm{tr}(AB) = \mathrm{tr}(BA)$ (cyclic shifting)

$$
\begin{align*}
(AB)_{ii} &= \sum_j A_{ij} B_{ji}, \\
\mathrm{tr}(AB) &= \sum_i \sum_j A_{ij} B_{ji}, \\
\mathrm{tr}(BA) &= \sum_j \sum_i B_{ji} A_{ij}.
\end{align*}
$$

1. Proving $\|A\|_F^2 = \mathrm{tr}(A^\top A)$

$$
\begin{align*}
[A^\top A]_{ii} &= \sum_j A_{ji} A_{ji}, \\
\mathrm{tr}(A^\top A) &= \sum_i \sum_j A_{ji} A_{ji} = \|A\|_F^2
\end{align*}
$$

### Von-Neumann's Trace Inequality

In 1937, Von-Neumann proved if A, B are complex n×n matrices with singular values

$$
a_1 \ge a_2 \ge \cdots \ge a_n,\quad b_1 \ge b_2 \ge \cdots \ge b_n
$$

Then

$$
|tr(AB)| <= \sum_i (a_i b_i)
$$

Maximum is achieved when A and B are diagonal:

$$
A = \operatorname{diag}(a_1, a_2, \dots, a_n),\quad
B = \operatorname{diag}(b_1, b_2, \dots, b_n),\quad
\operatorname{tr}(AB) = \sum_{i=1}^n a_i b_i.
$$

### Singular values of a rotation matrix are all 1

Let $R \in SO(3)$, i.e., $R^\top R = I$ and $\det R = 1$. The singular values $\{\sigma_i\}_{i=1}^3$ of $R$ are the square roots of the eigenvalues of $R^\top R$:

$$
\sigma_i = \sqrt{\lambda_i(R^\top R)}.
$$

Since $R^\top R = I$, all eigenvalues of $R^\top R$ are $1$. Therefore,

$$
\sigma_1 = \sigma_2 = \sigma_3 = 1.
$$

Equivalently, from the SVD $R = U\Sigma V^\top$ with $U, V \in SO(3)$ for an orthogonal matrix, we must have $\Sigma = I$, hence all singular values are $1$.

## Frobenius Norm

Frobenius norm = $\sum_i \sum_j (a_{ij} * a_{ij})$

E.g., a common task in lidar is if we have an estimate $R$ of an SO(3) matrix, we want to find the closest SO(3) matrix $X$ with the lowest Frobenius norm. That is:

1. Proving $\|X-R\|_F^2 = \|X\|_F^2 + \|R\|_F^2 - 2\mathrm{tr}(X^\top R)$ and deriving $\arg\max(\mathrm{tr}(X^\top R))$

$$
\begin{align*}
\|X-R\|_F^2 &= \mathrm{tr}((X-R)^\top(X-R)) \\
&= \mathrm{tr}(X^\top X - X^\top R - R^\top X + R^\top R) \\
&= \mathrm{tr}(X^\top X) - \mathrm{tr}(X^\top R) - \mathrm{tr}(R^\top X) + \mathrm{tr}(R^\top R) \\
&= \|X\|_F^2 - \mathrm{tr}(X^\top R) - \mathrm{tr}(X^\top R) + \|R\|_F^2 \\
&= \|X\|_F^2 + \|R\|_F^2 - 2\mathrm{tr}(X^\top R)
\end{align*}
$$

2. To minimize $\|X-R\|_F^2$, we need to maximize $\mathrm{tr}(X^\top R)$ since $\|X\|_F^2$ and $\|R\|_F^2$ are constants. Therefore:

$$\arg\min_R \|X-R\|_F^2 = \arg\max_X \mathrm{tr}(X^\top R)$$

3. Now, we can perform SVD on $R$:

$$R = U \Sigma V^\top$$

4. To find $X$, we define an intermediate variable:

$$Y = U^\top X V$$

5. Since $U$ and $V$ are orthonormal matrices, they are in $\mathrm{O}(3)$. Consequently, $Y$ is also in $\mathrm{O}(3)$. So now $\mathrm{tr}(X^\top R)$ becomes:

$$
\begin{align*}
\mathrm{tr}(X^\top R) &= \mathrm{tr}((UYV^\top)^\top (U \Sigma V^\top)) \\
&= \mathrm{tr}(VY^\top U^\top U \Sigma V^\top) \\
&= \mathrm{tr}(VY^\top \Sigma V^\top)
\end{align*}
$$

6. Because of the cyclic shifting property (see above):

$$
\begin{align*}
\mathrm{tr}(VY^\top \Sigma V^\top) &= \mathrm{tr}(Y^\top \Sigma V^\top V) \\
&= \mathrm{tr}(Y^\top \Sigma)
\end{align*}
$$

7. Choosing $Y$ and the determinant constraint

Let $R = U\,\Sigma\,V^\top$ be the SVD with $U, V \in \mathrm{O}(3)$ and $\Sigma = \operatorname{diag}(\sigma_1,\sigma_2,\sigma_3)$, $\sigma_1 \ge \sigma_2 \ge \sigma_3 \ge 0$. Define $Y := U^\top X V$. Then $Y \in \mathrm{O}(3)$ and

$$
\det(Y)
= \det(U^\top)\,\det(X)\,\det(V)
= \det(U)\,\det(V)\cdot 1
= \det(UV^\top) \in \{\pm 1\}.
$$

8. Moreover,

$$
\mathrm{tr}(X^\top R) = \mathrm{tr}(Y^\top \Sigma),
$$

9. Since maximum of $\mathrm{tr}(Y^\top \Sigma)$ is achieved when $Y$ and $\Sigma$ are diagonal. Since $\Sigma$ is diagonal already, we want $Y$ to be a diagonal SO(3) matrix. Therefore, maximize $\mathrm{tr}(Y^\top \Sigma)$ over $Y \in \mathrm{O}(3)$ subject to $\det(Y) = \det(UV^\top)$.

- If $\det(UV^\top) = 1$, the maximizer is $Y = I$, giving $\mathrm{tr}(Y^\top \Sigma) = \sigma_1 + \sigma_2 + \sigma_3$.
- If $\det(UV^\top) = -1$, the maximizer (under $\det(Y)=-1$) is $Y = \operatorname{diag}(1,1,-1)$, giving $\mathrm{tr}(Y^\top \Sigma) = \sigma_1 + \sigma_2 - \sigma_3$.

10. Thus, the optimizer for the original problem is

$$
X^* = U\,\operatorname{diag}\big(1,\,1,\,\det(UV^\top)\big)\,V^\top.
$$

```cpp
Eigen::Matrix3d nearestRotation(const Eigen::Matrix3d& R) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d Vt = svd.matrixV().transpose();
    Eigen::Matrix3d R_ortho = U * Vt;
    if (R_ortho.determinant() < 0.0) {
        Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
        S(2,2) = -1.0;
        R_ortho = U * S * Vt;
    }
    return R_ortho;
}
```
