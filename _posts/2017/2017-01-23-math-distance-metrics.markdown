---
layout: post
title: Math - Distance Metrics
subtitle: KL Divergence , Chi-Squared Similarity, Mahalanobis Distance
date: '2017-01-23 13:19'
header-img: "img/bg-walle.jpg"
tags:
    - Math
---

## Statistical Distance & Divergence Metrics

### Kullback-Leibler (KL) Divergence

Given two distributions, $p(x)$, and $q(x)$, denotes **how different $p(x)$ is from $q(x)$**, hence it further denotes **how much information will be lost when q(x) is used to represent p(x)**

$$
\begin{gather*}
D_{KL}(p(x) || q(x)) = \sum_X p(x) ln(\frac{p(x)}{q(x)})
\end{gather*}
$$

- KL Divergence is not a distance, because the $KL(x)$ from $p(x)$ to $q(x)$ usually is not the same as that from $q(x)$ to $p(x)$
- $KL(x) \ge 0$, when $p(x)=q(x)$, $KL(x)=0$

From counting, we find that $q(x_i)=0$ for a certain value $x_i$, technically,

$$
D_{KL}(p(x) | q(x)) = \sum_X p(x) ln(\frac{p(x)}{0}) = \inf
$$.

However, this could cause a lot of issues. instead, we can assume $q(x) = \epsilon = 10^{-3}$ in this case to avoid numerical errors

#### KL Divergence Can Never Be Negative

$$
\begin{aligned}
D_{\mathrm{KL}}(p(x) \,\|\, q(x)) 
&= \int p(x) \ln\left( \frac{p(x)}{q(x)} \right) \, dx \\
&= \int -p(x) \ln\left( \frac{q(x)}{p(x)} \right) \, dx \\
&\ge -\ln\left( \int p(x) \frac{q(x)}{p(x)} \, dx \right) \quad \text{(Jensen's inequality)} \\
&= -\ln\left( \int q(x) \, dx \right) \\
&= -\ln(1) = 0
\end{aligned}
$$

- Note: a pdf value at a point is **NOT** a probability. The probability here is 0. The PDF value could be larger or smaller than 1.

#### Special Case: `nn.CrossEntropy()`

When the target distribution $p(x)$ is an one-hot vector, the above formulation becomes [cross-entropy](../2022/2022-01-24-deep-learning-softmax-crossentropy.markdown):

$$
\begin{gather*}
D_{KL}(p(x) || q(x)) = - \sum_i y_i log(\hat{y_i})
\end{gather*}
$$

### Chi-Squared Similarity

Chi-Squared Similarity is often used to measure probability distributions of categoritcal data, such as histograms, counts, text data represented by term frequencies.

$$
\begin{gather*}
\chi^2(P,Q) = \frac{(P_i-Q_i)^2}{(P_i+Q_i)}
\end{gather*}
$$

Where $P_i$, $Q_i$ are bins for distributions $P$, $Q$. Denometer $P_i + Q_i$ brings a normalization effect, which considers different scales of the distributions.

## Mahalanobis Distance

The Mahalanobis distance is a measure of “how far” a point $x$ lies from the mean $\mu$ of a multivariate distribution, taking into account the scale (variance) and correlations of the data. 

$$
\begin{gather*}
\begin{aligned}
& D_M(x) = \sqrt{(x - \mu)^\top \Sigma^{-1} (x - \mu)}
\end{aligned}
\end{gather*}
$$

### Bhattacharyya Distance

TODO
