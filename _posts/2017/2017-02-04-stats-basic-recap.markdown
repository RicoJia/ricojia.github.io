---
layout: post
title: Math - Stats Basics Recap
subtitle: Basic Statistics Concepts, Regression, Distributions, Covariance & Correlation, Covariance Propagation, Variance Addition, Inverse-Variance Weighting, Bessel Correction, box filter
date: 2017-02-04 13:19
header-img: img/bg-material.jpg
tags:
  - Math
---

## Basic Statistics Concepts

### Standard Deviation and Variance

Variance of a distribution can be "biased" and "unbiased". A biased variance is to always underestimate the real bias.

$$
\begin{gather*}
\text{unbiased variance} = \frac{\sum (x - \bar{x})}{n-1}
\\
\text{biased variance} = \frac{\sum (x - \bar{x})}{n}
\end{gather*}
$$

The above unbiasing operation is caleld "Bessel correction"

An Isotropic covariance matrix is:

$$
\begin{gather*}
\begin{aligned}
& C = \lambda I
\end{aligned}
\end{gather*}
$$

### Stochastic Processes and Stationarity

A stochastic process is a collection of Random Variables over time. If a random variable is `X`, a process of it is `X(t)`. If the mean and variance of `X(t)` does not change, then loosely, this process is stationary.

#### Reasoning For Bessel Correction

- Population variance (or the true variance of the entire population) is calculated as:

$$
\begin{gather*}
\sigma^2 = \frac{1}{N} \sum_N (x - \mu)^2
\end{gather*}
$$

    - Where `N` is the whole popilation's size, $$\mu$$ is the population mean

- Sample variance:

$$
\begin{gather*}
s^2 = \frac{1}{n} \sum_n (x - \bar{x})^2
\end{gather*}
$$

    - Where `n` is the batch size, $$\bar{x}$$ is the batch mean

The sample variance has a slight bias because $$\bar{x}$$ is a random variable dependent on the sample. The population mean is slightly larger, so we divide by $$N-1$$ instead of $$N$$.

## Distributions

### Student's t-distribution

Student's t-distribution is similar to a Gaussian distribution, but with heavier tails and shorter peaks.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/e1b05729-6a45-4f89-bb39-3ecf4b3cb047" height="300" alt=""/>
    </figure>
</p>
</div>

TODO

## Covariance And Correlation

Given two random variables $$A$$, $$B$$

- Mean

$$
\begin{gather*}
\mu_A = \frac{1}{N} \sum_i A_i, \mu_B = \frac{1}{N} \sum_i B_i
\end{gather*}
$$

- Standard Deviation

$$
\begin{gather*}
\sigma_A = \sqrt{\frac{1}{N} \sum_i (A_i - \mu_A)^2},
\sigma_B = \sqrt{\frac{1}{N} \sum_i (B_i - \mu_B)^2}
\end{gather*}
$$

    - Standard Deviation indicates how spread out the data is from the mean. 

- Covariance

$$
\begin{gather*}
cov(AB) = \frac{1}{N} \sum_i (A_i - \mu_A)(B_i - \mu_B)
\end{gather*}
$$

    - Covariance indicates **the Joint variability of $$(A_i, B_i)$$ pairs together.** If a single pair of $$A_i$$, $$B_i$$ are both positive, you will get a positive value. If one of them is positive, one of them is negative, you will get a negative value. Altogether, they could indicate how related $$A$$ and $$B$$ are. 

- Correlation

$$
\begin{gather*}
corr(AB) = \frac{cov(AB)}{\sigma_A \sigma_B}
\end{gather*}
$$

    - Correlation is a standardized measure of "relatedness" between two random variables. It ranges from $$[-1, 1]$$. If $$A=kB$$ after mean normalization, then correlation will be a perfect 1

### Covariance Propagation Through a Nonlinear Function

A random variable isn't "linear" or "nonlinear"; the function that transforms it is. Given a random vector `x` with known mean and covariance, pushed through a nonlinear function `f`:

$$
\begin{gather*}
\mathbf{x}\sim(\boldsymbol{\mu}_x,\Sigma_x), \quad \mathbf{y}=f(\mathbf{x})
\end{gather*}
$$

how do we get the mean and covariance of `y`?

#### Start With a Scalar

Take `y = 3x` with `σ_x = 2`. A change of 1 in `x` changes `y` by 3, so the uncertainty is amplified by 3 too: `σ_y = 6`. In terms of variance, with the derivative `J = 3`:

$$
\begin{gather*}
\sigma_y^2=3^2\sigma_x^2 = J\sigma_x^2J^T
\end{gather*}
$$

The transpose does nothing for a scalar, but this is already Jacobian-based covariance propagation.

#### Multiple Inputs and Outputs: The Jacobian

With several inputs and outputs, the Jacobian collects every partial derivative. It is a **sensitivity matrix**: entry `(i, j)` says how much output `i` moves when input `j` moves by 1. For example:

$$
\begin{gather*}
\begin{aligned}
y_1&=2x_1+x_2
\\ y_2&=x_1+3x_2
\end{aligned}
\quad \Rightarrow \quad
J=
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1}&\frac{\partial y_1}{\partial x_2}
\\ \frac{\partial y_2}{\partial x_1}&\frac{\partial y_2}{\partial x_2}
\end{bmatrix}
=
\begin{bmatrix}2&1\\1&3\end{bmatrix}
\end{gather*}
$$

#### Derivation of `J Σ J^T`

Linearize `f` around the mean with a first-order Taylor expansion, where `J` is the Jacobian evaluated at `μ_x`:

$$
\begin{gather*}
f(\mathbf{x})\approx f(\boldsymbol{\mu}_x)+J(\mathbf{x}-\boldsymbol{\mu}_x)
\end{gather*}
$$

So a small input error `δx` maps to an output error `δy ≈ J δx`. Covariance is the expected outer product of errors, `Σ_x = E[δx δx^T]`, so:

$$
\begin{gather*}
\begin{aligned}
\Sigma_y
&=E[\delta\mathbf{y}\delta\mathbf{y}^T]
\\ &\approx E[(J\delta\mathbf{x})(J\delta\mathbf{x})^T]
\\ &=J\, E[\delta\mathbf{x}\delta\mathbf{x}^T]\, J^T
\end{aligned}
\end{gather*}
$$

$$
\begin{gather*}
\boxed{\Sigma_y\approx J\Sigma_xJ^T}, \quad \boxed{\boldsymbol{\mu}_y\approx f(\boldsymbol{\mu}_x)}
\end{gather*}
$$

- `J` turns small input errors into output errors.
- `Σ_x` describes the input error distribution.
- `J Σ_x J^T` is the resulting output covariance.

This is a **first-order** approximation: it replaces `f` with its tangent at `μ_x`. It is good for small errors. When the input uncertainty is large relative to the curvature of `f`, both `J Σ J^T` and `f(μ_x)` become inaccurate (the true mean of `y` is generally not `f(μ_x)`).

#### Example: INS Uncertainty of a Sonar Target

A vehicle with pose `[x, y, θ]` detects a mine 20 m along its forward axis. The mine's global position and the Jacobian with respect to the pose are:

$$
\begin{gather*}
\begin{aligned}
p_x&=x+20\cos\theta
\\ p_y&=y+20\sin\theta
\end{aligned}
\quad \Rightarrow \quad
J_{\text{nav}}=
\begin{bmatrix}
1&0&-20\sin\theta
\\ 0&1&20\cos\theta
\end{bmatrix}
\overset{\theta=0}{=}
\begin{bmatrix}
1&0&0
\\ 0&1&20
\end{bmatrix}
\end{gather*}
$$

That 20 is the lever arm: a 1 m vehicle position error gives a 1 m target error, but a 0.01 rad heading error gives about `20 × 0.01 = 0.2` m of transverse target error.

Suppose the INS has 1 m standard deviation on each position axis and 0.1 rad on heading:

$$
\begin{gather*}
\Sigma_{\text{nav}}=\operatorname{diag}(1,\ 1,\ 0.01)
\end{gather*}
$$

$$
\begin{gather*}
\Sigma_{\text{geo,nav}}=
J_{\text{nav}}\Sigma_{\text{nav}}J_{\text{nav}}^T
=
\begin{bmatrix} 1&0&0\\ 0&1&20 \end{bmatrix}
\begin{bmatrix} 1&0&0\\ 0&1&0\\ 0&0&0.01 \end{bmatrix}
\begin{bmatrix} 1&0\\ 0&1\\ 0&20 \end{bmatrix}
=
\begin{bmatrix} 1&0\\ 0&1 + 20^2 \cdot 0.01 \end{bmatrix}
=
\boxed{\begin{bmatrix} 1&0\\ 0&5 \end{bmatrix}}
\end{gather*}
$$

So from the INS alone, `σ_px = 1` m and `σ_py = √5 ≈ 2.24` m: heading uncertainty dominates the transverse error. This is only the navigation contribution; the sonar measurement's own covariance, propagated through its own Jacobian, adds on top of it if the two errors are independent (see the next sections).

### Variance Is Not a Gaussian Property

Variance measures how much a random variable deviates from its mean. For a zero-mean random variable:

$$
\begin{gather*}
\operatorname{Var}(X)=E[X^2]=\sigma_X^2
\end{gather*}
$$

More generally:

$$
\begin{gather*}
\boxed{\operatorname{Var}(X)=E[X^2]-E[X]^2}
\end{gather*}
$$

**Variance applies to any probability distribution with a finite second moment**, not just Gaussians. For example, rolling a fair six-sided die is a discrete uniform distribution: each outcome has the same probability.

```
Probability
  1/6 |  █   █   █   █   █   █
      |  █   █   █   █   █   █
      |  █   █   █   █   █   █
    0 +-------------------------
         1   2   3   4   5   6
                  X
```

Its mean and variance are:

$$
\begin{gather*}
E[X]=3.5, \quad \operatorname{Var}(X)=\frac{35}{12}\approx 2.92
\end{gather*}
$$

We love Gaussian distributions because **they are described entirely by their mean and covariance**. For non-Gaussian distributions, the mean and covariance do not necessarily capture the entire distribution.

### Adding Two Independent Random Variables

Another property that is not specific to Gaussian distributions: **the variances of two independent random variables add together**.

Suppose `X` and `Y` are independent and both zero-mean. Define `Z = X + Y`. Then:

$$
\begin{gather*}
\begin{aligned}
\operatorname{Var}(Z)
&=E[(X+Y)^2]
\\ &=E[X^2+2XY+Y^2]
\\ &=E[X^2]+2E[XY]+E[Y^2]
\end{aligned}
\end{gather*}
$$

Because `X` and `Y` are independent, `E[XY] = E[X]E[Y] = 0`. Therefore:

$$
\begin{gather*}
\boxed{\operatorname{Var}(X+Y)=\sigma_X^2+\sigma_Y^2}
\end{gather*}
$$

This is the **additive property of variance for independent random variables**. The same applies to covariance matrices of independent random vectors:

$$
\begin{gather*}
\boxed{\Sigma_{X+Y}=\Sigma_X+\Sigma_Y}
\end{gather*}
$$

If the variables are correlated, **we must also account for their cross-covariance.**

For example, when estimating the global position of a sonar detection, we combine the sonar measurement uncertainty and the INS state-estimation uncertainty. Assuming the two errors are independent, their covariances add together after being transformed into the same coordinate frame.

### Combining Multiple Independent Measurements: Inverse-Variance Weighting

When combining multiple measurements, we are still working with independent random variables. However, **they are measurements of the same quantity**.

Suppose two sensors estimate the position of the same object, `X_1` and `X_2`. Each measurement has its own mean and variance:

$$
\begin{gather*}
(\mu_1,\sigma_1^2),\qquad(\mu_2,\sigma_2^2)
\end{gather*}
$$

A simple average is not optimal when the measurements have different uncertainties. Instead, we want a **weighted average** that gives more weight to measurements with smaller variance.

**Step 1:** Assign inverse-variance weights. The normalized weight of measurement `i` is:

$$
\begin{gather*}
\boxed{w_i=\frac{1/\sigma_i^2}{\sum_j 1/\sigma_j^2}}, \quad \sum_i w_i=1
\end{gather*}
$$

**Step 2:** Calculate the weighted mean. A measurement with smaller variance contributes more:

$$
\begin{gather*}
\boxed{\mu_{\text{fused}}=\sum_i w_i\mu_i}
\end{gather*}
$$

**Step 3:** Calculate the combined variance. Because the measurements are independent, the variance of their weighted sum is:

$$
\begin{gather*}
\sigma_{\text{fused}}^2 = w_1^2\sigma_1^2+w_2^2\sigma_2^2+\cdots
\end{gather*}
$$

**Notice that the weights are squared.** This follows from `Var(aX) = a² Var(X)`. Substituting the inverse-variance weights and simplifying:

$$
\begin{gather*}
\begin{aligned}
\sigma_{\text{fused}}^2
&= \sum_i \left( \frac{1/\sigma_i^2}{\sum_j1/\sigma_j^2} \right)^2\sigma_i^2
\\ &= \frac{\sum_i1/\sigma_i^2}{\left(\sum_j1/\sigma_j^2\right)^2}
\\ &= \boxed{\frac{1}{\sum_i1/\sigma_i^2}}
\end{aligned}
\end{gather*}
$$

This is the **inverse-variance fusion formula**. It gives the minimum-variance linear unbiased estimate when measurements are unbiased, independent, and have known finite variances. It does not require Gaussian distributions.

#### Connection To The Kalman Filter

The **Kalman filter measurement update is essentially inverse-variance weighted averaging**. The Kalman gain determines how much weight to assign to the new measurement versus the prior estimate, with lower-variance estimates receiving higher weights. For a scalar state, the Kalman gain is exactly the normalized inverse-variance weight of the measurement:

$$
\begin{gather*}
K=\frac{1/\sigma_z^2}{1/\sigma_p^2+1/\sigma_z^2}
\end{gather*}
$$

The updated estimate is:

$$
\begin{gather*}
\mu_{\text{new}}=(1-K)\mu_p+Kz
\end{gather*}
$$

The updated variance follows the same inverse-variance fusion rule:

$$
\begin{gather*}
\sigma_{\text{new}}^2=\left(\frac{1}{\sigma_p^2}+\frac{1}{\sigma_z^2}\right)^{-1}
\end{gather*}
$$

This applies to independent, unbiased estimates of the same scalar quantity. In multidimensional Kalman filters, the same principle extends to covariance matrices.

## Regression

Regression in statistics means "estimating the relationship model between dependent variables and independent variables." For example, linear regression models the relationship of `Y` and its independent variables, `x1, x2 ...`.

```
Y=β0​+β1​x1​+β2​x2​+⋯+βn​xn​+ϵ
```

Transformer is "autoregressive". It's regressive because it tries to model the relationship between output and input sequences. It's "auto" because the output sequence depends on the previous output.

## Random Process

A random process $$R(t)$$ is basically a collection of random variables that vary along time. The random variables' mean and standard deviations may or may not change. If they don't change, we call the random process **stationary**

### Gaussian Random Process

A Gaussain Random Process is

$$
\begin{gather*}
\begin{aligned}
& R(t) \sim \mathcal{gp}(m(t), k(t, t') )
\end{aligned}
\end{gather*}
$$

Where the mean function of the Random Process is $$m(t)$$, and the **covariance function** $$k(t, t')$$ could change over time, too.

$$
\begin{gather*}
\begin{aligned}
& m(t) = E[R(t)]
\\
& k(t, t') = E[(R(t) - m(t))(R(t') - m(t'))]
\end{aligned}
\end{gather*}
$$

One special case is white Gaussian Random noise:

$$
\begin{gather*}
\begin{aligned}
& R(t) \sim \mathcal{gp}(0, \delta(t - t') \sigma^2)
\end{aligned}
\end{gather*}
$$

the covariance $$\sigma$$ does not change across time. Between different times, `t, t'`, there's no correlation between them, and they are independent.
$$\delta(t - t')$$ is "Dirac Delta Distribution."  It's a probability distribution, where everywhere is 0 except for at time `t'`. Also, $$\int_{-\infty}^{\infty} \delta(t-t')f(t) = f(t')$$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/e6b0a7b9-8ee2-4f6b-a85f-78c1534110aa" height="300" alt=""/>
    </figure>
</p>
</div>

### Power Spectral Density

In signal procesisng, if we view a signal `x(t)` as a random process, then we can find its power across all frequencies. This is called "power spectral density" (PSD).

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/aec3b75d-2846-4e33-b57a-c4930049a028" height="300" alt=""/>
        <figcaption><a href="https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html">Source: Mathworks</a></figcaption>
    </figure>
</p>
</div>

It's defined as the Fourier Transform of the auto-correlation of the signal function at time difference $$\tau$$. The autocorrelation is:

$$
\begin{gather*}
\begin{aligned}
& R_{xx}(\tau) = E[x(t)x(t + \tau)]
\end{aligned}
\end{gather*}
$$

So if the signal is periodic with period of $$\tau$$, $$R_{xx}(n\tau)$$ would peak. The PSD $$S_{xx}(f)$$ is then the Fourier Transform of the autocorrelation across all time differences, $$\tau$$:

$$
\begin{gather*}
\begin{aligned}
& S_{xx}(f) = F( R_{xx}(\tau)) = \int_{-\infty}^{\infty} R_{xx}(\tau) e^{-j2\pi f \tau} d\tau
\end{aligned}
\end{gather*}
$$

For white Gaussian noise, the PSD is a **constant** $$\sigma^2$$:

$$
\begin{gather*}
\begin{aligned}
& S_{xx}(f) = F( R_{xx}(\tau)) = \int_{-\infty}^{\infty} \sigma^2 \delta(\tau) e^{-j2\pi f \tau} d\tau = \sigma^2
\end{aligned}
\end{gather*}
$$

### Wiener Process

Wiener Process is a.k.a Brownian Motion. It's non-stationary

$$
\begin{gather*}
\begin{aligned}
& W(t + \Delta t) = W(t) + \Delta W
\\
& \Delta W \sim \mathcal{N}(0, \Delta t)
\end{aligned}
\end{gather*}
$$

Its mean is 0, but variance is t (so it increases)

$$
\begin{gather*}
\begin{aligned}
& E[W(t)] = 0
\\ & Var[W(t)] = t
\end{aligned}
\end{gather*}
$$

## Box Filter

a box filter is a same-size local average, one output value per input pixel, not a downsample
