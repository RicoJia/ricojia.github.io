---
layout: post
title: Deep Learning - Optimizations
date: '2023-02-04 13:19'
subtitle: Batch Gradient Descent, RMSDrop, Adam Optimization
comments: true
tags:
    - Deep Learning
---

## A Neuron And Batch Gradient Descent
A Neuron, has multiple inputs and a single output. First it gets the weighted sum of all inputs, then feeds it into an "activation function". Below, the activation function $\sigma(z)$ is a "sigma function"

$$
\begin{gather*}
z = \sum_{i}^{n} w_ix_i = w_0x_0 + w_1 x_1 ...
\\
y = \sigma(z) = \frac{1}{1 + e^{-z}}
\end{gather*}
$$


<p align="center">
<img src="https://github.com/RicoJia/The-Dream-Robot/assets/39393023/49e6c48c-65e3-44c5-b130-db4138440499" height="300" width="width"/>
<figcaption align="center">Image Source: Stackoverflow</figcaption>
</p>

So for a single input, x (`nx1` vector), and its corresponding groudtruth value y, we can get its prediction $\hat{y}$ and cost $J$. In our example, J is cross-entropy, which has its minimum cost when $y = \hat{y}$

$$
\begin{gather*}
\hat{y} = \sigma(\sum_{i}^{n} w_ix_i)
\\
J = -(ylog(\hat{y}) + (1-y)log(1-\hat{y}))
\end{gather*}
$$

Then, we can apply gradient descent to update parameters, $w$, which is the **goal of deep learning**. The gradient is represented as $\nabla{J}$, which is the partial derivative w.r.t all parameters in vector $w$. Since gradient gives the steepest **ascent**, we want to update $w$ with the negative of that.

$$
\begin{gather*}
\nabla{J} = [\frac{\partial{J}}{\partial{w_0}}, \frac{\partial{J}}{\partial{w_1}} ...]
\\
w = w - \lambda\nabla{J}
\end{gather*}
$$

For an individual partial derivative value,

$$
\begin{gather*}
\frac{\partial{J}}{\partial{w_i}} = \frac{\partial{J}}{\partial{\hat{y}}} \frac{\partial{\hat{y}}}{\partial{z}} \frac{\partial{z}}{\partial{w_i}}
\\ 
\frac{\partial{J}}{\partial{\hat{y}}} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}
\\
\frac{\partial{\hat{y}}}{\partial{z}} = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} (1-\frac{1}{1 + e^{-z}}) 
= \hat{y}(1-\hat{y})
\\
\frac{\partial{z}}{\partial{w_i}} = x_i
\\
=>
\\
\frac{\partial{J}}{\partial{w_i}} = (\hat{y} - y)x_i
\frac{\partial{J}}{\partial{w_i}} = (\hat{y} - y)
\\
\end{gather*}
$$

So from the single input $x$, we update $w$ with

$$
\begin{gather*}
w = w - \lambda\nabla{J} = w - \lambda(\hat{y} - y)x
\\
b = b - \lambda\nabla{J} = w - \lambda(\hat{y} - y)
\end{gather*}
$$

**Now if we have a batch inputs, that is $x^{(0)} ... x^{(m)}$**

$$
\begin{gather*}

J = -\frac{1}{m} \sum_{m}^{M} (y^{(m)} log(\hat{y}^{(m)}) + (1-y^{(m)})log(1-\hat{y}^{(m)}))
\\
w = w - \sum_{m}^{M}\lambda\nabla{J^{(m)}} = w - \frac{1}{m} \sum_{m}^{M} \lambda(\hat{y}^{(m)} - y^{(m)})x^{(m)}
\\
b = b - \lambda\nabla{J} = b - \lambda(\hat{y} - y)
\end{gather*}
$$
