---
layout: post
title: Math - 3D Point Distribution Metrics
subtitle: Median Absolute Deviation (mad), Auto-Correlated Function (ACF)
date: 2017-01-23 13:19
header-img: img/bg-walle.jpg
tags:
  - Math
---

## std/MAD

Two ways to measure "spread":

- std $=\sqrt{\mathbb E[X^2]}$ — squares everything, so one big outlier dominates. Measures the extremes.
- MAD $=\text{median}|X|$ — half the values are above, half below. One outlier changes nothing. Measures the typical.

`std/MAD` tells us: how far out are the extremes, relative to pure Gaussian? For a Gaussian, $\text{median}|X| = 0.6745\sigma$, so

$$r_{\text{Gauss}} = \frac{\sigma}{0.6745\sigma} = 1.4826$$

Any $r > 1.4826$ means **heavier-tailed** than Gaussian. Real data at $r = 23.2$ is 15.7× more tail-heavy than Gaussian.

### What is ACF?

Autocorrelation function. For a pixel and its neighbour $k$ steps away:

$$\rho(k) = \frac{\mathbb E[X_i X_{i+k}]}{\mathbb E[X_i^2]}$$

$\rho(0)=1$ always. $\rho(k)\to 0$ as $k$ grows. The shape of that decay is the texture of the noise.

- $\rho(k)=0$ for all $k\neq0$ → pure salt-and-pepper grain
- $\rho$ decays slowly → soft blobby mottling

The key theorem: if you build a field by blurring white noise with a kernel $h$, the field's ACF is the kernel's own autocorrelation:

$$\rho(k) = \frac{\sum_j h[j],h[j+k]}{\sum_j h[j]^2}$$

For our Gaussian kernel with $\sigma=0.82$, the autocorrelation of a Gaussian is another Gaussian of width $\sigma\sqrt2$:

$$\rho(k) = e^{-k^2/(4\sigma^2)}$$

Theory matches our measurement to three decimals. So we understand our own field exactly — and the problem is unambiguous: $k^2$ in the exponent kills it. Real decays like $a^k$ (linear in $k$), we decay like $e^{-k^2}$. By lag 4 we're 30× too small. The fix, and why it's exact. Use a one-sided exponential kernel $h[j]=a^j$ for $j\ge0$:
$$\sum_{j\ge0} a^j a^{j+k} = a^k\sum_{j\ge0}a^{2j} = \frac{a^k}{1-a^2} \quad\Longrightarrow\quad \rho(k) = a^k$$

Exponential ACF by construction, no fitting. Set $a_{\text{row}}=0.651$, $a_{\text{col}}=0.489$. Honest limit: look at the last column. Exponential overshoots at high lag (0.276 vs 0.213 at lag 3). It's ~3× better than Gaussian, not perfect. Real sits between the two. Worth taking the improvement and re-measuring.

Gotcha: $h$ is one-sided, so it's a causal filter. *padding*=K//2 assumes a symmetric kernel and will shift the whole field by $K/2$ pixels. It needs explicit left-padding only.
