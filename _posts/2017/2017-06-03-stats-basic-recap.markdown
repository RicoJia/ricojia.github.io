---
layout: post
title: Math - Stats Basics Recap
subtitle: Basic Statistics Concepts, Regression, Distributions, Covariance & Correlation
date: '2017-06-03 13:19'
header-img: "img/bg-material.jpg"
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

#### Reasoning For Bessel Correction

- Population variance (or the true variance of the entire population) is calculated as:

$$
\begin{gather*}
\sigma^2 = \frac{1}{N} \sum_N (x - \mu)^2
\end{gather*}
$$
    - Where `N` is the whole popilation's size, $\mu$ is the population mean

- Sample variance:

$$
\begin{gather*}
s^2 = \frac{1}{n} \sum_n (x - \bar{x})^2
\end{gather*}
$$
    - Where `n` is the batch size, $\bar{x}$ is the batch mean

The sample variance has a slight bias because $\bar{x}$ is a random variable dependent on the sample. The population mean is slightly larger, so we divide by $N-1$ instead of $N$.

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

## Regression

Regression in statistics means "estimating the relationship model between dependent variables and independent variables." For example, linear regression models the relationship of `Y` and its independent variables, `x1, x2 ...`.

```
Y=β0​+β1​x1​+β2​x2​+⋯+βn​xn​+ϵ
```

Transformer is "autoregressive". It's regressive because it tries to model the relationship between output and input sequences. It's "auto" because the output sequence depends on the previous output.
