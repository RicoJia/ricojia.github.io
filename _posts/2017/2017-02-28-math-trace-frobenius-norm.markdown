---
layout: post
title: Math - Trace, Determinant, Frobenius Norm
date: '2017-02-28 13:19'
subtitle: Trace Properties and Proofs, Von-Neumann's Trace Inequality
comments: true
tags:
    - Math
---

## Determinant

- $$\det(AB) = \det(A)\,\det(B)$$
- For an orthogonal matrix $$Q$$ (i.e., $$Q^\top Q = I$$), we have $$\det(Q) = \pm 1$$:

$$
\begin{aligned}
Q^\top Q &= I \\
\Rightarrow\; \det(Q^\top Q) &= \det(I) = 1 \\
\Rightarrow\; \det(Q^\top)\,\det(Q) &= 1 \\
\Rightarrow\; \det(Q)\,\det(Q) &= 1 \quad (\text{since } \det(Q^\top) = \det(Q)) \\
\Rightarrow\; (\det Q)^2 &= 1 \\
\Rightarrow\; \det Q &= \pm 1.
\end{aligned}
$$

In particular, if $$Q \in SO(n)$$ (special orthogonal group), then $$\det Q = 1$$.

## Trace Properties

The trace of a square matrix is just the sum of its diagonal entries: $$\operatorname{tr}(A) = \sum_i A_{ii}$$. It looks simple, but a handful of its properties show up all over robotics, so let's prove them one by one.

### Linearity and Transpose

These all follow directly from the definition, because the trace only ever looks at the diagonal:

$$
\begin{aligned}
\operatorname{tr}(A + B) &= \sum_i (A_{ii} + B_{ii}) = \operatorname{tr}(A) + \operatorname{tr}(B), \\
\operatorname{tr}(cA) &= \sum_i c A_{ii} = c\,\operatorname{tr}(A), \\
\operatorname{tr}(A^\top) &= \operatorname{tr}(A) \quad (\text{transposing does not move diagonal entries}).
\end{aligned}
$$

### Cyclic Shifting

$$\operatorname{tr}(AB) = \operatorname{tr}(BA)$$. This holds even when $$AB$$ and $$BA$$ have different sizes, say $$A$$ is $$m \times n$$ and $$B$$ is $$n \times m$$. To see why, write out the diagonal of $$AB$$:

$$
\begin{aligned}
(AB)_{ii} &= \sum_j A_{ij} B_{ji}, \\
\operatorname{tr}(AB) &= \sum_i \sum_j A_{ij} B_{ji}, \\
\operatorname{tr}(BA) &= \sum_j \sum_i B_{ji} A_{ij}.
\end{aligned}
$$

The two double sums contain exactly the same terms, just added in a different order, so they are equal. Applying this repeatedly, we can rotate a longer product around:

$$
\operatorname{tr}(ABC) = \operatorname{tr}(CAB) = \operatorname{tr}(BCA)
$$

Note that only cyclic rotations are allowed. In general, $$\operatorname{tr}(ABC) \ne \operatorname{tr}(ACB)$$.

### Invariance Under Change of Basis

For any invertible $$P$$, cyclic shifting gives:

$$
\operatorname{tr}(P^{-1}AP) = \operatorname{tr}(A P P^{-1}) = \operatorname{tr}(A)
$$

So the trace doesn't care which coordinate system we use. That's what allows it to equal something basis-independent, like the sum of eigenvalues below.

### Trace Equals the Sum of Eigenvalues

$$
\operatorname{tr}(A) = \sum_i \lambda_i
$$

**Quick proof for a diagonalizable matrix** (this includes every symmetric matrix). Write $$A = V \Lambda V^{-1}$$. Then by the change-of-basis invariance above,

$$
\operatorname{tr}(A) = \operatorname{tr}(V \Lambda V^{-1}) = \operatorname{tr}(\Lambda) = \sum_i \lambda_i
$$

**General proof**, using the characteristic polynomial. The eigenvalues are its roots, so we can factor it as

$$
\det(\lambda I - A) = \prod_{i=1}^n (\lambda - \lambda_i) = \lambda^n - \Big(\sum_i \lambda_i\Big)\lambda^{n-1} + \cdots
$$

Now let's expand the same determinant directly. A determinant is a sum of products, where each product picks one entry from every row and every column. Only the product of the diagonal entries, $$\prod_i (\lambda - A_{ii})$$, can reach $$\lambda^{n-1}$$. Any other product has to skip at least two diagonal entries, so it has at most $$n-2$$ factors of $$\lambda$$. Expanding the diagonal product:

$$
\prod_i (\lambda - A_{ii}) = \lambda^n - \Big(\sum_i A_{ii}\Big)\lambda^{n-1} + \cdots
$$

Matching the $$\lambda^{n-1}$$ coefficients of the two expansions gives $$\sum_i A_{ii} = \sum_i \lambda_i$$. This works for every square matrix, as long as we count eigenvalues with multiplicity (and allow them to be complex). As a bonus, comparing the constant terms the same way gives $$\det(A) = \prod_i \lambda_i$$.

### Trace of an Outer Product

$$
\operatorname{tr}(\mathbf{x}\mathbf{y}^\top) = \sum_i x_i y_i = \mathbf{x}^\top\mathbf{y},
\qquad
\operatorname{tr}(\mathbf{x}\mathbf{x}^\top) = \|\mathbf{x}\|^2
$$

The diagonal entries of $$\mathbf{x}\mathbf{y}^\top$$ are $$x_i y_i$$, and summing them gives the dot product. Another way to see it: by cyclic shifting, $$\operatorname{tr}(\mathbf{x}\mathbf{y}^\top) = \operatorname{tr}(\mathbf{y}^\top\mathbf{x})$$, and $$\mathbf{y}^\top\mathbf{x}$$ is a $$1 \times 1$$ matrix, which is its own trace.

### Trace of a Squared Skew-Symmetric Matrix

Let $$\mathbf{v} \in \mathbb{R}^3$$, and let $$[\mathbf{v}]_\times$$ (also written $$\mathbf{v}^\wedge$$) be its skew-symmetric matrix, so that $$[\mathbf{v}]_\times \mathbf{w} = \mathbf{v} \times \mathbf{w}$$. Then

$$
\operatorname{tr}\left([\mathbf{v}]_\times^\top [\mathbf{v}]_\times\right) = 2\|\mathbf{v}\|^2,
\qquad
\operatorname{tr}\left([\mathbf{v}]_\times^2\right) = -2\|\mathbf{v}\|^2
$$

First, we find what $$[\mathbf{v}]_\times^\top [\mathbf{v}]_\times$$ actually is. Since $$[\mathbf{v}]_\times^\top = -[\mathbf{v}]_\times$$, applying it to any vector $$\mathbf{w}$$ and using the triple product expansion $$\mathbf{a} \times (\mathbf{b} \times \mathbf{c}) = \mathbf{b}(\mathbf{a}\cdot\mathbf{c}) - \mathbf{c}(\mathbf{a}\cdot\mathbf{b})$$ gives

$$
[\mathbf{v}]_\times^\top [\mathbf{v}]_\times \mathbf{w}
= -\mathbf{v} \times (\mathbf{v} \times \mathbf{w})
= -\mathbf{v}(\mathbf{v}\cdot\mathbf{w}) + \mathbf{w}\|\mathbf{v}\|^2
\quad\Rightarrow\quad
[\mathbf{v}]_\times^\top [\mathbf{v}]_\times = \|\mathbf{v}\|^2 I - \mathbf{v}\mathbf{v}^\top
$$

Then, taking the trace with linearity and the outer-product result above:

$$
\operatorname{tr}\left(\|\mathbf{v}\|^2 I - \mathbf{v}\mathbf{v}^\top\right) = 3\|\mathbf{v}\|^2 - \|\mathbf{v}\|^2 = 2\|\mathbf{v}\|^2
$$

Since $$[\mathbf{v}]_\times^2 = -[\mathbf{v}]_\times^\top[\mathbf{v}]_\times$$, the second form just flips the sign. This identity shows up in [the quaternion-to-angle derivation](https://ricojia.github.io/2024/03/13/quaternion/) and in [the rotation block of an ICP Hessian](https://ricojia.github.io/2026/05/14/Hessian-Degeneracy-Test/).

### Trace of a Rotation Matrix

A rotation by angle $$\theta$$ about a unit axis $$\mathbf{a}$$ has trace

$$
\operatorname{tr}(R) = 1 + 2\cos\theta
$$

To see this, start from Rodrigues' formula, $$R = \cos\theta\, I + (1 - \cos\theta)\,\mathbf{a}\mathbf{a}^\top + \sin\theta\,[\mathbf{a}]_\times$$, and take the trace term by term. We have $$\operatorname{tr}(I) = 3$$ and $$\operatorname{tr}(\mathbf{a}\mathbf{a}^\top) = \|\mathbf{a}\|^2 = 1$$. The last term vanishes, because a skew-symmetric matrix has zeros on its diagonal, so $$\operatorname{tr}([\mathbf{a}]_\times) = 0$$. Putting it together:

$$
\operatorname{tr}(R) = 3\cos\theta + (1 - \cos\theta) = 1 + 2\cos\theta
\quad\Rightarrow\quad
\theta = \cos^{-1}\left(\frac{\operatorname{tr}(R) - 1}{2}\right)
$$

This is exactly how we [recover the rotation angle from a rotation matrix](https://ricojia.github.io/2024/03/10/robotics-foundamentals-rotations/).

### Frobenius Norm as a Trace

$$
\|A\|_F^2 = \operatorname{tr}(A^\top A)
$$

The $$i$$-th diagonal entry of $$A^\top A$$ is the squared length of the $$i$$-th column of $$A$$. Summing over all columns adds up every squared entry of $$A$$:

$$
\begin{aligned}
[A^\top A]_{ii} &= \sum_j A_{ji} A_{ji}, \\
\operatorname{tr}(A^\top A) &= \sum_i \sum_j A_{ji} A_{ji} = \|A\|_F^2
\end{aligned}
$$

### Von-Neumann's Trace Inequality

In 1937, Von-Neumann proved that if $$A, B$$ are complex $$n \times n$$ matrices with singular values

$$
a_1 \ge a_2 \ge \cdots \ge a_n,\quad b_1 \ge b_2 \ge \cdots \ge b_n
$$

then

$$
\lvert \operatorname{tr}(AB) \rvert \le \sum_i a_i b_i
$$

The bound is reached, for example, when $$A$$ and $$B$$ are both diagonal with their singular values sorted the same way:

$$
A = \operatorname{diag}(a_1, a_2, \dots, a_n),\quad
B = \operatorname{diag}(b_1, b_2, \dots, b_n),\quad
\operatorname{tr}(AB) = \sum_{i=1}^n a_i b_i.
$$

### Singular Values of a Rotation Matrix Are All 1

Let $$R \in SO(3)$$, i.e., $$R^\top R = I$$ and $$\det R = 1$$. The singular values $$\{\sigma_i\}_{i=1}^3$$ of $$R$$ are the square roots of the eigenvalues of $$R^\top R$$:

$$
\sigma_i = \sqrt{\lambda_i(R^\top R)}.
$$

Since $$R^\top R = I$$, all eigenvalues of $$R^\top R$$ are $$1$$. Therefore,

$$
\sigma_1 = \sigma_2 = \sigma_3 = 1.
$$

Equivalently, in the SVD $$R = U\Sigma V^\top$$ with orthogonal $$U$$ and $$V$$, we must have $$\Sigma = I$$, so all singular values are $$1$$.

## Frobenius Norm

The Frobenius norm is the square root of the sum of all squared entries: $$\|A\|_F = \sqrt{\sum_i \sum_j a_{ij}^2}$$.

E.g., a common task in lidar is: we have an estimate $$R$$ of a rotation matrix that has drifted away from $$SO(3)$$, and we want the closest true rotation $$X \in SO(3)$$, i.e. the one with the lowest $$\|X - R\|_F$$. Here's how to find it.

**Step 1: turn the norm into a trace.** Using $$\|A\|_F^2 = \operatorname{tr}(A^\top A)$$ and linearity,

$$
\begin{aligned}
\|X-R\|_F^2 &= \operatorname{tr}((X-R)^\top(X-R)) \\
&= \operatorname{tr}(X^\top X - X^\top R - R^\top X + R^\top R) \\
&= \operatorname{tr}(X^\top X) - \operatorname{tr}(X^\top R) - \operatorname{tr}(R^\top X) + \operatorname{tr}(R^\top R) \\
&= \|X\|_F^2 - \operatorname{tr}(X^\top R) - \operatorname{tr}(X^\top R) + \|R\|_F^2 \\
&= \|X\|_F^2 + \|R\|_F^2 - 2\operatorname{tr}(X^\top R)
\end{aligned}
$$

where the fourth line uses $$\operatorname{tr}(R^\top X) = \operatorname{tr}((X^\top R)^\top) = \operatorname{tr}(X^\top R)$$.

**Step 2: minimizing the norm means maximizing the trace.** $$\|R\|_F^2$$ is fixed, and $$\|X\|_F^2 = \operatorname{tr}(X^\top X) = \operatorname{tr}(I) = 3$$ for any rotation $$X$$. So both are constants, and

$$
\arg\min_X \|X-R\|_F^2 = \arg\max_X \operatorname{tr}(X^\top R)
$$

**Step 3: simplify with the SVD.** Take the SVD $$R = U\,\Sigma\,V^\top$$, with $$U, V \in \mathrm{O}(3)$$ and $$\Sigma = \operatorname{diag}(\sigma_1,\sigma_2,\sigma_3)$$, $$\sigma_1 \ge \sigma_2 \ge \sigma_3 \ge 0$$. Then define an intermediate variable

$$
Y = U^\top X V \quad\Leftrightarrow\quad X = U Y V^\top
$$

$$Y$$ is a product of orthogonal matrices, so it is orthogonal too ($$Y \in \mathrm{O}(3)$$). Substituting, and then using cyclic shifting to move $$V^\top$$ to the front:

$$
\begin{aligned}
\operatorname{tr}(X^\top R) &= \operatorname{tr}((UYV^\top)^\top (U \Sigma V^\top)) \\
&= \operatorname{tr}(VY^\top U^\top U \Sigma V^\top) \\
&= \operatorname{tr}(VY^\top \Sigma V^\top) \\
&= \operatorname{tr}(Y^\top \Sigma V^\top V) \\
&= \operatorname{tr}(Y^\top \Sigma)
\end{aligned}
$$

**Step 4: watch the determinant.** $$X$$ must be a proper rotation, $$\det(X) = 1$$, which pins down the determinant of $$Y$$:

$$
\det(Y)
= \det(U^\top)\,\det(X)\,\det(V)
= \det(U)\,\det(V)\cdot 1
= \det(UV^\top) \in \{\pm 1\}.
$$

**Step 5: pick the best $$Y$$.** By Von-Neumann's trace inequality, $$\operatorname{tr}(Y^\top \Sigma)$$ is largest when $$Y$$ is diagonal, since $$\Sigma$$ already is. A diagonal orthogonal matrix has $$\pm 1$$ on its diagonal, so we maximize over those, subject to $$\det(Y) = \det(UV^\top)$$:

- If $$\det(UV^\top) = 1$$, the maximizer is $$Y = I$$, giving $$\operatorname{tr}(Y^\top \Sigma) = \sigma_1 + \sigma_2 + \sigma_3$$.
- If $$\det(UV^\top) = -1$$, we need an odd number of $$-1$$'s, and the cheapest place to put one is on the smallest singular value: $$Y = \operatorname{diag}(1,1,-1)$$, giving $$\operatorname{tr}(Y^\top \Sigma) = \sigma_1 + \sigma_2 - \sigma_3$$.

**Step 6: map back.** Since $$X = U Y V^\top$$, the closest rotation is

$$
X^* = U\,\operatorname{diag}\big(1,\,1,\,\det(UV^\top)\big)\,V^\top.
$$

In code:

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
