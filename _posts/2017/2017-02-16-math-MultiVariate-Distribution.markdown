---
layout: post
title: Math - Multivariate Normal Distribution
date: '2017-02-13 13:19'
subtitle: PDF, Linear Transformation, Covariance Matrix
comments: true
tags:
    - Math
---

## PDF

"Variate" is basically a synonym to "random variable". If we have a vector of random variables and we want to find their joint distributution, then we call this "multivariate" distribution.

If all variates have PDFs (probability density function) of Gaussian distributions, then the joint distribution is:

$$
\begin{gather*}
\begin{aligned}
& p = \frac{1}{(2 \pi)^{n/2}|\Sigma|^{1/2}} exp(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu))
\end{aligned}
\end{gather*}
$$

Where:

- n is the number of variates
- $\Sigma$ is the covariance matrix.
- $|\Sigma|$ is the determinant of the covariance matrix.
- $\mu$ is an n-vector $[\mu_1 ... \mu_n]$

Example:

Given:

$$
\begin{gather*}
\begin{aligned}
& \mu = [0, 1]
\\
& \Sigma =
\begin{bmatrix}
1 & 0.5 \\
0.5 & 2
\end{bmatrix}
\end{aligned}
\end{gather*}
$$

We know:

$$
\begin{gather*}
\begin{aligned}
& |\Sigma| = det(\Sigma) = 2 - 0.25 = 1.75
\\
& \Sigma^{-1} = \begin{bmatrix}
1.14 & -0.29 \\
-0.29 & 0.57
\end{bmatrix}

\end{aligned}
\end{gather*}
$$

So at point `x = [1,1]`, we can compute the pdf:

$$
\begin{gather*}
\begin{aligned}
& x - \mu = [1, 0]
\\
& (x - \mu)^T \Sigma^{-1} (x - \mu) = 1.14
\\
& \text{pdf([1,1])} =  \frac{1}{(2 \pi)1.75^{1/2}} exp(1.14)
\end{aligned}
\end{gather*}
$$

As can be seen, the final pdf value is always a number. Here's an illustration from [Wikipedia](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#/media/File:MultivariateNormal.png):

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/f47c173d-3dcd-4095-a05f-bbbd49bf5159" height="300" alt=""/>
       </figure>
    </p>
</div>

### A Word On The Covariance Matrix

An unbiased covariance matrix is $\frac{1}{M-1} \sum_N(x_m - \mu_m)(x_m - \mu_m)^T$. In MLE (Maximum Likelihood Estimate), it's a biased one: $\frac{1}{N} \sum_M(x_m - \mu_m)(x_m - \mu_m)^T$

- `m` is the number of observations of the joint probability: $x_m = [x1, x2 ...]$

Meanwhile, the covariance matrix is a symmetric, positive semi-definite matrix.

### How to estimate a multivariate Gaussian distribution's Covariance Matrix?

$$
\begin{gather*}
\begin{aligned}
& -log(p) \propto ((x - \mu)^T \Sigma^{-1} (x - \mu))
\\
& \text{Hessian: } D^2(-log(p)) = \Sigma^{-1}
\end{aligned}
\end{gather*}
$$

## Linear Transformation

If we have two joint distributions: $y = Ax + b$, where x has a mean vector $\mu_x$, covariance matrix $\Sigma_x$, then:

$$
\begin{gather*}
\begin{aligned}
& \mu_y = \mu_x + b
\\
& \Sigma_y = A^T \Sigma_x A
\end{aligned}
\end{gather*}
$$
