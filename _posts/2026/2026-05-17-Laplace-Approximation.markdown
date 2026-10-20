---
layout: post
title: Laplace Approximation
date: 2026-05-17 13:19
subtitle: From Hessian Curvature to Gaussian Uncertainty
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine-Learning
---

The **Laplace approximation** approximates a complicated probability distribution with a Gaussian, using the location of its maximum and the curvature around that maximum. It is the reason the Hessian from the [Hessian Degeneracy Test](https://ricojia.github.io/2026/05/14/Hessian-Degeneracy-Test/) can be read as an inverse covariance.

## 1. Start with a Non-Gaussian Distribution

Suppose we want to estimate a parameter `x`, and its probability distribution is:

$$
p(x)=\frac{1}{Z}\exp\left[-(x-2)^2-0.1(x-2)^4\right]
$$

Here:

- `x` is our unknown parameter.
- `Z` is a normalization constant that makes the total probability equal to 1.
- The distribution is not Gaussian because of the fourth-order term `0.1(x−2)^4`.

Define the **negative log-probability**, ignoring the constant `log Z`:

$$
L(x)=(x-2)^2+0.1(x-2)^4
$$

Minimizing `L(x)` is equivalent to maximizing `p(x)`.

## 2. Find the Maximum (MAP Estimate)

Set the derivative to zero:

$$
\frac{dL}{dx}=2(x-2)+0.4(x-2)^3=(x-2)\left[2+0.4(x-2)^2\right]=0
\qquad\Rightarrow\qquad
\boxed{x^*=2}
$$

The bracket is always positive, so `x* = 2` is the only stationary point. But knowing `x* = 2` doesn't tell us how certain we are about that estimate. For uncertainty, we need the **curvature** of `L`, i.e. its second derivative.

Note that `x*` is the **MAP** (maximum a posteriori) estimate: the parameter value that maximizes the posterior, which is proportional to the likelihood of the data times the prior. With a flat prior, it is the maximum-likelihood estimate.

## 3. Calculate the Hessian

In one dimension, the Hessian is simply the second derivative. Evaluated at the optimum:

$$
\frac{d^2L}{dx^2}=2+1.2(x-2)^2
\qquad\Rightarrow\qquad
H=L''(2)=2
$$

What does this number tell us?

- A **large** Hessian means the negative log-probability rises steeply away from the optimum, producing a **narrow** probability peak.
- A **small** Hessian means the loss changes slowly, producing a **broader** peak.

Therefore, `H` contains information about uncertainty.

## 4. Apply a Taylor Expansion

This is where the Laplace approximation comes from. The second-order Taylor expansion of `L` around `x*` is below. Since `x*` minimizes `L`, the first-order term vanishes, `L'(x*) = 0`:

$$
L(x)\approx L(x^*)+\underbrace{L'(x^*)(x-x^*)}_{=0}+\frac12H(x-x^*)^2
$$

Substituting our numbers, `L(2) = 0` and `H = 2`:

$$
L(x)\approx 0+\frac12(2)(x-2)^2=\boxed{(x-2)^2}
$$

This approximates the original function near the optimum, `x = 2`.

## 5. The Result Is a Gaussian

Going back to the probability distribution:

$$
p(x)=\frac{1}{Z}\exp\left[-L(x)\right]
\approx C\exp\left[-\frac12H(x-x^*)^2\right]
$$

This is a Gaussian! Compare it with:

$$
\mathcal N(x;\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right]
$$

The exponents have the same form:

$$
\underbrace{\frac12H(x-x^*)^2}_{\text{Laplace}} \quad\longleftrightarrow\quad \underbrace{\frac{(x-\mu)^2}{2\sigma^2}}_{\text{Gaussian}}
\qquad\Rightarrow\qquad
\boxed{\mu=x^*,\qquad \sigma^2=\frac{1}{H}}
$$

In our example, `σ² = 1/2`, so `σ ≈ 0.71`. The quartic term makes the true distribution fall off faster than this Gaussian away from `x = 2`, so here the approximation slightly overestimates the spread.

## 6. Extending to Robotics

Now imagine estimating a 2D robot position. Suppose optimization converges to `x*`, and the Hessian of the negative log-posterior at that solution is `H`:

$$
\mathbf x= \begin{bmatrix} t_x\\t_y \end{bmatrix},
\qquad
\mathbf x^*= \begin{bmatrix} 3\\5 \end{bmatrix},
\qquad
H= \begin{bmatrix} 100&0\\ 0&4 \end{bmatrix}
$$

The multidimensional Taylor expansion gives:

$$
L(\mathbf x)\approx L(\mathbf x^*)+ \frac12(\mathbf x-\mathbf x^*)^TH(\mathbf x-\mathbf x^*)
\qquad\Rightarrow\qquad
p(\mathbf x)\propto \exp\left[ -\frac12(\mathbf x-\mathbf x^*)^TH(\mathbf x-\mathbf x^*) \right]
$$

Compare that with the multivariate Gaussian:

$$
p(\mathbf x)\propto \exp\left[ -\frac12(\mathbf x-\boldsymbol\mu)^T \Sigma^{-1} (\mathbf x-\boldsymbol\mu) \right]
\qquad\Rightarrow\qquad
\boxed{\boldsymbol\mu=\mathbf x^*,\qquad \Sigma=H^{-1}}
$$

So:

$$
\Sigma=H^{-1}= \begin{bmatrix} 0.01&0\\ 0&0.25 \end{bmatrix}
\qquad\Rightarrow\qquad
\sigma_x=\sqrt{0.01}=0.1\text{ m},
\qquad
\sigma_y=\sqrt{0.25}=0.5\text{ m}
$$

The large curvature in `x` (100) means `x` is well constrained; the small curvature in `y` (4) means `y` is five times more uncertain. This is very useful in SLAM and nonlinear optimization because it lets us estimate uncertainty directly from the Hessian.

![](https://i.postimg.cc/J02xXZRH/Screenshot-from-2026-10-19-13-46-28.png)

The inner dashed ellipse is the 1σ Mahalanobis contour; the outer solid ellipse is 2σ. They show greater uncertainty in `y` than in `x`. In 2D, the 1σ ellipse contains about 39.3% of the Gaussian probability mass (`1 − e^{−1/2}`), and the 2σ ellipse about 86.5% (`1 − e^{−2}`).

## 7. From Covariance to One Degeneracy Score: Magnusson's $$Q_H$$ {#magnusson-score}

Once $$\Sigma=H^{-1}$$ is a covariance, its worst direction gives a single confidence number for a registration. Martin Magnusson proposed this in his 2009 PhD thesis on the 3D Normal-Distributions Transform [1]:

$$
\boxed{Q_H=\sqrt{\lambda_{\max}(H^{-1})}}
$$

The eigenvalues of $$\Sigma$$ are the variances along its principal directions, so $$\lambda_{\max}(H^{-1})$$ is the largest variance. **The square root converts it into a standard deviation**, which has the same unit as the state. $$Q_H$$ is therefore the standard deviation of the least certain direction, and **a larger $$Q_H$$ means a less trustworthy registration.**

In the robot example above, $$Q_H=\max(\sigma_x,\sigma_y)=\sqrt{0.25}=0.5\text{ m}$$.

### $$Q_H$$ only needs the smallest Hessian eigenvalue

For a symmetric positive-definite $$H=V\Lambda V^T$$, the inverse is $$H^{-1}=V\Lambda^{-1}V^T$$, so [every eigenvalue becomes its reciprocal](https://ricojia.github.io/2017/01/15/eigen-value-decomp/#zero-eigenvalues-invertibility-and-condition-number). The largest eigenvalue of $$H^{-1}$$ therefore comes from the smallest eigenvalue of $$H$$:

$$
\lambda_{\max}(H^{-1})=\frac{1}{\lambda_{\min}(H)}
\qquad\Rightarrow\qquad
\boxed{Q_H=\frac{1}{\sqrt{\lambda_{\min}(H)}}}
$$

Check with the example: $$\lambda_{\min}(H)=4$$, so $$Q_H=1/\sqrt{4}=0.5\text{ m}$$, the same as before. No matrix inverse is needed.

| $$\lambda_{\min}(H)$$ | $$Q_H$$ |
|---|---|
| 100 | 0.1 |
| 4 | 0.5 |
| 1 | 1 |
| 0.04 | 5 |
| 0.01 | 10 |
| 0.0001 | 100 |

As $$\lambda_{\min}(H)\rightarrow0^+$$, $$Q_H\rightarrow\infty$$. For an exactly singular Hessian, the uncertainty along its null space is unbounded unless a prior is added.

### Limitations

- **Mixed units.** A 6-DoF pose mixes meters and radians, so $$\lambda_{\min}$$ and $$Q_H$$ change with the unit choice even when the geometry does not. Compute it on the scaled Hessian $$\mathbf{H}_s$$ from [Hessian Degeneracy Test, Part 4](https://ricojia.github.io/2026/05/14/Hessian-Degeneracy-Test/#part-4---additional-handling-for-rotation-scale).
- **Curvature is local.** A steep minimum does not mean the correct minimum. Repetitive geometry can create several sharp local minima, so $$Q_H$$ measures local confidence, not global correctness.
- **One scalar hides the direction.** $$Q_H$$ reports how uncertain the worst direction is, not which direction it is. Two scenes can share a $$Q_H$$ with different degenerate directions. The eigenvector of $$\lambda_{\min}$$ carries that information; see [Hessian Degeneracy Test, Parts 2–3](https://ricojia.github.io/2026/05/14/Hessian-Degeneracy-Test/#part-2---why-hessian-eigenvalues-can-show-degeneracy).

[1] Magnusson, M. (2009). [_The Three-Dimensional Normal-Distributions Transform — An Efficient Representation for Registration, Surface Analysis, and Loop Detection._](https://www.diva-portal.org/smash/get/diva2:276162/FULLTEXT02.pdf) PhD thesis, Örebro University. Section 6.3.1, Equation 6.28.

## 8. Summary

The entire Laplace approximation fits in three steps:

$$
\begin{aligned}
p(\mathbf x\mid\mathbf z) &\propto \exp[-L(\mathbf x)]\\[6pt]
L(\mathbf x) &\approx L(\mathbf x^*)+ \frac12(\mathbf x-\mathbf x^*)^TH(\mathbf x-\mathbf x^*)\\[6pt]
p(\mathbf x\mid\mathbf z) &\approx \mathcal N(\mathbf x^*,H^{-1})
\end{aligned}
$$

- **Requirements:** a sufficiently **smooth loss**, a **local minimum** (stationary point), and a positive-definite Hessian. It works best when the probability mass is concentrated around one dominant mode.
- **Central insight:** we are not claiming the original distribution is Gaussian. **Near its maximum, its negative log-probability looks approximately quadratic, so the distribution looks locally Gaussian.** That is the mathematical connection between optimization curvature and probabilistic uncertainty.
- **Base assumption:** the inverse Hessian is a statistically meaningful covariance only if the loss is a properly scaled negative log-posterior (or an appropriate likelihood-based approximation). For example, residuals must be weighted by their noise covariance, and the cost must not carry an extra factor: if a solver minimizes `Σ rᵀ Σ⁻¹ r` without the `½`, its Hessian is `2×` the information matrix.
