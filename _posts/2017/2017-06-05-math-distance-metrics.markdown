---
layout: post
title: Math - Distance Metrics
subtitle: KL Divergence , Chi-Squared Similarity
date: '2017-06-05 13:19'
header-img: "img/bg-walle.jpg"
tags:
    - Math
---

## Statistical Distance & Divergence Metrics

### Kullback-Lebler (KL) Divergence

Given two distributions, $p(x)$, and $q(x)$, denotes **how different $p(x)$ is from $q(x)$**, hence it further denotes **how much information will be lost when q(x) is used to represent p(x)**

$$
\begin{gather*}
D_{KL}(p(x) || q(x)) = \sum_X p(x) \frac{p(x)}{q(x)}
\end{gather*}
$$

- KL Divergence is not a distance, because the $KL(x)$ from $p(x)$ to $q(x)$ usually is not the same as that from $q(x)$ to $p(x)$
- $KL(x) \ge 0$, when $p(x)=q(x)$, $KL(x)=0$
- If from counting, we find that $q(x_i)=0$ for a certain value $x_i$, technically, $D_{KL}(p(x) || q(x)) = \sum_X p(x) \frac{p(x)}{0} = \inf$. However, this could cause a lot of issues. instead, we can assume $q(x) = \epsilon = 10^{-3}$ in this case to avoid numerical errors

### Chi-Squared Similarity

Chi-Squared Similarity is often used to measure probability distributions of categoritcal data, such as histograms, counts, text data represented by term frequencies.

$$
\begin{gather*}
\chi^2(P,Q) = \frac{(P_i-Q_i)^2}{(P_i+Q_i)}
\end{gather*}
$$

Where $P_i$, $Q_i$ are bins for distributions $P$, $Q$. Denometer $P_i + Q_i$ brings a normalization effect, which considers different scales of the distributions.

### Bhattacharyya Distance

TODO