---
layout: post
title: Math - MAP(Maximum A-Posteriori) and MLE (Maximum Likelihood Estimate)
date: '2017-02-13 13:19'
subtitle: Likelihood vs Probabilities
comments: true
tags:
    - Math
---

## Likelihood VS Probabilities

Probabilities describe the chances of discrete, mutually-exclusive possible states. These chances should sum up to 1. Likelihoods describe the chances or the plausibility of non-mutually exclusive, and potentially infinite and continuous hypotheses. For example, we have a robot that could only be in positions `{1, 2, 3}`. There is a landmark from a distance. The robot has a **probability distribution** of its own position: `P(X) = {p(x=1), p(x=2), p(x=3)}`. These probabilities should sum up to 1. In the mean time, the robot has a likelihood function of the landmark observation: `L(Z|X)`. Part of the likelihood function `L(z|x = 1)` could look like a bell curve. The landmark could be in multiple highly-likely positions, and the likelihood need not sum up to 1.

## MLE (最大似然) vs MAP （最大后验估计）

Maximum Likelihood Estimate focuses on finding the most likely state variables `x` that maximizes the observation data likelihood, `z`. So this is purely data-driven. It does NOT consider prior information:

$$
\begin{gather*}
\begin{aligned}
& argmax(P(z|x))
\end{aligned}
\end{gather*}
$$

MAP not only considers MLE, but also considers the prior states `x`. It is more stale than MLE and can work better in a Bayesian Filter framework. When the data is limited, MAP might be better. When observation and single state variable data are abundant, the prior's influence diminishes

$$
\begin{gather*}
\begin{aligned}
& argmax(P(x|z)) \alpha P(z|x) P(x)
\end{aligned}
\end{gather*}
$$

In real life, since we always deal with Gaussian Distributions, we use the log trick on probabilities to make computaiton easier.

### Example of MLE

This example is inspired by [this post](https://sassafras13.github.io/MLEvsMAP/).

Assume now my robot is at an unknown location, $\mu$. A landmark is at `x=(0)`. The robot has 3 measurements: 5m, 8m, 9m. We assume that the likelihoods of these measurements follow a Gaussian noise distribution: $P(z|x) = \frac{1}{\sigma\sqrt{2 \pi}} e^{-\frac{(x - \mu)^2}{2 \sigma^2}}$

Therefore, the joint likelihood of having these measurements at the location $\mu$ is:

$$
\begin{gather*}
\begin{aligned}
& P(z_1, z_2, z_3|x) = \frac{1}{\sigma\sqrt{2 \pi}} e^{-\frac{(5 - \mu)^2}{2 \sigma^2}} \cdot \frac{1}{\sigma\sqrt{2 \pi}} e^{-\frac{(8 - \mu)^2}{2 \sigma^2}} \cdot \frac{1}{\sigma\sqrt{2 \pi}} e^{-\frac{(9 - \mu)^2}{2 \sigma^2}}
\end{aligned}
\end{gather*}
$$

Now, we are going to find $\mu$ such that this joint likelihood is the smallest. We can do that by taking its partial derivative w.r.t $\mu$, then set it to 0. For the ease of computation, we do the log trick:

$$
\begin{gather*}
\begin{aligned}
& f = ln(P(z_1, z_2, z_3|x)) = 3 ln(\frac{1}{\sigma\sqrt{2 \pi}}) - {\frac{(5 - \mu)^2}{2 \sigma^2} + \frac{(8 - \mu)^2}{2 \sigma^2} + \frac{(9 - \mu)^2}{2 \sigma^2}}
\\
& \rightarrow \frac{\partial f}{\partial \mu} = \frac{\mu -5 + \mu - 8 + \mu - 9}{\sigma^2} = 0
\\
& \mu = \frac{22}{3}
\end{aligned}
\end{gather*}
$$
