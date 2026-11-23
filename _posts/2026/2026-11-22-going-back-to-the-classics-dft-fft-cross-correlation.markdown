---
layout: post
title: "[Going Back To The Classics] DFT, FFT, Cross-Correlation"
date: 2026-11-22 19:30
subtitle: DFT, Inverse DFT, FFT Even-Odd Split, Twiddle Factors, Circular Cross-Correlation
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Signal Processing
---

## 1. DFT: Time Domain → Frequency Domain

The Fourier transform says any time-domain signal can be decomposed into waves of different frequencies.

![](https://s3.amazonaws.com/cdn.freshdesk.com/data/helpdesk/attachments/production/150133475609/original/Venn1vJO237DSLhupLg8KUG3Ps387QY1mg.png?1760021938)

Let's walk through the DFT with a simple example.

Assume we have a time-domain signal sampled at 4 time steps (illustrative values, rather than exact values from the image):

$$
x[t] = \{4, 2, 0, 2\}
$$

Now, assume we want **4 frequency bins**:

$$
\begin{aligned}
X[k=0] &= 8 \\
X[k=1] &= 4 \\
X[k=2] &= 0 \\
X[k=3] &= 4
\end{aligned}
$$

The DFT formula is:

$$
X[k] = \sum_{t=0}^{N-1} x[t] e^{-i 2\pi k t / N}
$$

Where:

- $N = 4$: number of time samples
- $t$: time index
- $k$: frequency bin index
- $X[k]$: frequency-domain coefficient, which can be a complex number
- $f_s$: sampling rate in Hz

Bin $k$ corresponds to the actual frequency:

$$
f_k = \frac{k f_s}{N}
$$

and sample $t$ corresponds to the actual time:

$$
t_{\text{seconds}} = \frac{t}{f_s}
$$

For example, for frequency bin $k = 0$:

$$
X[0] = 4 + 2 + 0 + 2 = 8
$$

For $k = 1$:

$$
X[1] = 4 + 2(-i) + 0 + 2(i) = 4
$$

We repeat the same process for all 4 frequency bins and get:

$$
\boxed{X[k] = \{8, 4, 0, 4\}}
$$

- $X[0] = 8$ is the zero-frequency (DC) component. Dividing by 4 gives the signal's average value of 2.
- $X[1]$ and $X[3]$ describe a pair of positive and negative frequencies that together reconstruct a real cosine wave. Writing out their basis waves:

$$
\begin{aligned}
k = 1&: \quad e^{-i 2\pi t / 4} = e^{-i\pi t/2} \\
k = 3&: \quad e^{-i 6\pi t / 4} = e^{-i 3\pi t/2} = e^{i\pi t/2}
\end{aligned}
$$

Bins $k$ and $N - k$, at opposite ends of $[0, N)$, are the same wave rotating in opposite directions, $e^{-i\phi}$ and $e^{i\phi}$. They are complex conjugates, so for a real signal $X[N-k] = \overline{X[k]}$, and their sum is a real cosine.

There could be infinitely many possible frequencies in a continuous signal. However, for a DFT of $N = 4$ samples, we have exactly 4 **unique frequency** bins: you can write out that $X[k+4] = X[k]$, so the bins are periodic. The sampling rate determines their actual frequencies in Hz.

---

## 2. Inverse DFT: Frequency Domain → Time Domain

The inverse DFT recovers the original time-domain values:

$$
x[t] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{i 2\pi k t / N}
$$

We know:

$$
X[k] = \{8, 4, 0, 4\}
$$

For $t = 0$:

$$
x[0] = \frac{1}{4}(8 + 4 + 0 + 4) = 4
$$

For $t = 1$:

$$
x[1] = \frac{1}{4}(8 + 4i + 0 - 4i) = 2
$$

Similarly:

$$
x[2] = 0, \qquad x[3] = 2
$$

Therefore:

$$
\boxed{x[t] = \{4, 2, 0, 2\}}
$$

We recovered the original signal exactly!

## 3. FFT (Fast Fourier Transform)

FFT is simply a faster algorithm to calculate the DFT.

Recall the DFT formula:

$$
X[k] = \sum_{t=0}^{N-1} x[t] e^{-i 2\pi k t / N}
$$

To calculate one frequency bin, we need to sum over all $N$ time samples. Because we have $N$ frequency bins, the direct DFT requires approximately $O(N^2)$ operations.

FFT reduces this to:

$$
\boxed{O(N \log_2 N)}
$$

### 3.1 Split into Even and Odd Indices

Split the summation into even and odd indices, $t = 2m$ and $t = 2m + 1$:

$$
X[k] = \underbrace{\sum_{m=0}^{N/2-1} x[2m] \, e^{-i 2\pi k (2m) / N}}_{\text{even contribution}}
+ \underbrace{\sum_{m=0}^{N/2-1} x[2m+1] \, e^{-i 2\pi k (2m+1) / N}}_{\text{odd contribution}}
$$

The odd exponent splits as $e^{-i 2\pi k (2m+1) / N} = e^{-i 2\pi k m / (N/2)} \, e^{-i 2\pi k / N}$. Therefore, the odd contribution is:

$$
\sum_{m=0}^{N/2-1} x[2m+1] \, e^{-i 2\pi k m / (N/2)} \, e^{-i 2\pi k / N}
$$

The factor $e^{-i 2\pi k / N}$ does not depend on $m$, so we can pull it outside the summation:

$$
\underbrace{e^{-i 2\pi k / N}}_{W_N^k}
\underbrace{\sum_{m=0}^{N/2-1} x[2m+1] \, e^{-i 2\pi k m / (N/2)}}_{O[k]}
= W_N^k O[k]
$$

The even contribution is simply $E[k]$. Both are $N/2$-point DFTs:

$$
\boxed{
\begin{aligned}
E[k] &= \sum_{m=0}^{N/2-1} x[2m] \, e^{-i 2\pi k m / (N/2)} \\
O[k] &= \sum_{m=0}^{N/2-1} x[2m+1] \, e^{-i 2\pi k m / (N/2)}
\end{aligned}
}
$$


Therefore:

$$
X[k] = E[k] + W_N^k O[k]
$$

For the upper half of the frequency bins, $E$ and $O$ repeat with period $N/2$, and the twiddle factor changes sign:

$$
W_N^{k + N/2} = -W_N^k
$$

Therefore:

$$
\boxed{
\begin{aligned}
X[k] &= E[k] + W_N^k O[k] \\
X[k + N/2] &= E[k] - W_N^k O[k]
\end{aligned}
}
$$

Note that $E[k]$ and $O[k]$ are two independent, smaller DFTs, so we can apply the same algorithm to each of them recursively. We keep splitting each DFT into even and odd samples until only one sample remains. A 1-point DFT is simply the sample itself. Then, we combine the results from the bottom up using the twiddle factors to recover the original DFT. For example:

$$
N = 8 \rightarrow 4 \rightarrow 2 \rightarrow 1
$$

### 3.2 Using Our Example

Original signal:

$$
x[t] = \{4, 2, 0, 2\}
$$

Split into:

$$
x_{\text{even}} = \{4, 0\}, \qquad x_{\text{odd}} = \{2, 2\}
$$

Since both have only two samples, their DFTs are particularly simple:

$$
\begin{aligned}
E[0] &= 4 + 0 = 4 \\
E[1] &= 4 - 0 = 4 \\[6pt]
O[0] &= 2 + 2 = 4 \\
O[1] &= 2 - 2 = 0
\end{aligned}
$$

Therefore:

$$
\boxed{E[k] = \{4, 4\}, \quad O[k] = \{4, 0\}}
$$

Now we can combine these using the twiddle factor:

$$
\boxed{
\begin{aligned}
X[k] &= E[k] + W_N^k O[k] \\
X[k + N/2] &= E[k] - W_N^k O[k]
\end{aligned}
}
$$

---

## 4. Cross-Correlation Theorem

Cross-correlation measures how similar two signals are when we shift one relative to the other. We can calculate it by multiplying the two signals in the frequency domain, then applying the inverse DFT. This is called the **cross-correlation theorem**, and it is one of the most useful applications of the DFT.

Here we use circular cross-correlation, meaning indices wrap around $N$ samples:

$$
R_{xy}[\ell] = \sum_{t=0}^{N-1} \overline{x[t]} \, y[(t + \ell) \bmod N]
$$

$R_{xy}[\ell]$ is the correlation score at shift $\ell$. For real signals, $\overline{x[t]} = x[t]$.

The theorem is:

$$
\boxed{R_{xy} = \operatorname{IDFT}\left(\overline{X} \, Y\right)}
$$

where $X = \operatorname{DFT}(x)$, $Y = \operatorname{DFT}(y)$, and $\overline{X[k]}$ is the complex conjugate of $X[k]$.

### 4.1 Derivation

Take the DFT of the correlation:

$$
\operatorname{DFT}(R_{xy})[k] = \sum_{\ell} \sum_{t} \overline{x[t]} \, y[t + \ell] \, e^{-i 2\pi k \ell / N}
$$

Now substitute:

$$
m = t + \ell, \qquad \ell = m - t
$$

The exponential becomes:

$$
e^{-i 2\pi k (m - t) / N} = e^{-i 2\pi k m / N} \, e^{i 2\pi k t / N}
$$

So we can separate the two summations:

$$
\operatorname{DFT}(R_{xy})[k] =
\underbrace{\sum_{t} \overline{x[t]} \, e^{i 2\pi k t / N}}_{\overline{X[k]}}
\underbrace{\sum_{m} y[m] \, e^{-i 2\pi k m / N}}_{Y[k]}
$$

Therefore:

$$
\boxed{\operatorname{DFT}(R_{xy})[k] = \overline{X[k]} \, Y[k]}
$$

And applying the inverse DFT gives:

$$
\boxed{R_{xy} = \operatorname{IDFT}\left(\overline{X} \, Y\right)}
$$
