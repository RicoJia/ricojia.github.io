---
layout: post
title: Math - Stats Basics Recap
subtitle: Distributions, Covariance & Correlation
date: '2017-06-03 13:19'
header-img: "img/bg-material.jpg"
tags:
    - Math
---

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

Given two random variables $A$, $B$

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

    - Covariance indicates **the Joint variability of $(A_i, B_i)$ pairs together.** If a single pair of $A_i$, $B_i$ are both positive, you will get a positive value. If one of them is positive, one of them is negative, you will get a negative value. Altogether, they could indicate how related $A$ and $B$ are. 

- Correlation

$$
\begin{gather*}
corr(AB) = \frac{cov(AB)}{\sigma_A \sigma_B}
\end{gather*}
$$
    - Correlation is a standardized measure of "relatedness" between two random variables. It ranges from $[-1, 1]$. If $A=kB$ after mean normalization, then correlation will be a perfect 1
