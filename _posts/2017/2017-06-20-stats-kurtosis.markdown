---
layout: post
title: [Statistics] Kurtosis - Simple Measure of Tail Heaviness
subtitle:
date: '2017-06-20 13:19'
header-img: "img/bg-material.jpg"
tags:
    - Statistics
---

## Introduction

Kurtosis is a statistic that tells us how strongly a distribution is influenced by unusually large values. For sensor residuals, that is useful because two datasets can have the same standard deviation but very different numbers of large errors.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://www.statisticshowto.com/wp-content/uploads/2016/05/heavy-tailed-300x178.png" height="300" alt=""/>
    </figure>
</p>
</div>

For a zero-mean variable $x$, kurtosis is

$$
\frac{E[x^4]}{E[x^2]^2}.
$$

So kurtosis is not mainly asking: "How large is the noise"? It is asking: "How heavy are the tails relative to the normal noise scale?"

A Gaussian distribution has kurtosis $3$.

## Small numerical example

Consider these residuals:

$$
x =
{-1,-1,1,1}.
$$

First compute the second moment:

$$
\frac{1+1+1+1}{4}
$$

Now compute the fourth moment:

$$
\frac{1+1+1+1}{4}
$$

Therefore, Kurtosis is

$$
\frac{1}{1^2}
$$

Now consider a second set:

$$
x =
{-0.5,-0.5,-0.5,1.5}.
$$

Its values are more uneven because one value is much larger than the others. The second moment is

$$
0.75.
$$

The fourth moment is

$$
1.3125.
$$

Therefore,

$$
\frac{1.3125}{0.75^2} =

\frac{1.3125}{0.5625}
\approx
2.33.
$$

So the second distribution has much higher kurtosis:

$$
1
\rightarrow
2.33.
$$

The reason is that **the fourth power strongly emphasizes large values.**

For example,

$$
1.5^2 = 2.25,
$$

but

$$
1.5^4 = 5.0625.
$$

The larger residual therefore contributes much more strongly to the fourth moment.

## Why this matters for synthetic sensor data

Suppose real range residuals occasionally contain large errors, but the synthetic generator mostly produces small Gaussian-like noise.

The generated data might match the real standard deviation:

$$
\sigma_{\text{gen}}
\approx
\sigma_{\text{real}},
$$

**while still producing too few extreme residuals**. Kurtosis can detect this difference.  A kurtosis-related loss could therefore encourage

$$
\text{kurtosis}{\text{gen}}
\rightarrow
\text{kurtosis}{\text{real}}.
$$

However, kurtosis uses the fourth power,

$$
x^4,
$$

so very large outliers can dominate it.  That makes kurtosis useful for measuring tail behavior, but also potentially sensitive to a small number of extreme samples. This is closely related to using

$$
\frac{L_4}{L_2}
$$

in a differentiable distribution loss. Since

$$
\left(E|x|^4\right)^{1/4}
$$

and

$$
\left(E|x|^2\right)^{1/2},
$$

we have

$$
\frac{E|x|^4}{E|x|^2}.
$$

For a zero-mean residual, this is essentially the kurtosis. So $L_4/L_2$ can be interpreted as a smooth, normalized measure of tail heaviness, closely connected to kurtosis.
