---
layout: post
title: Deep Learning - Gradient Checking
date: '2022-01-04 13:19'
subtitle: First Step To Debugging A Neural Net
comments: true
tags:
    - Deep Learning
---

## How To Do Gradient Checking

In calculus, we all learned that a derivative is defined as:

$$
\begin{gather*}
f'(\theta) = \frac{f(\theta + \epsilon) - f(\theta)}{\epsilon}
\end{gather*}
$$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/929472a6-edff-4fcc-9365-345a982dead7" height="300" alt=""/>
        <figcaption>Source: Andrew Ng's Deep Learning Class</figcaption>
    </figure>
</p>
</div>

In a neural net, the gradient of a parameter $w$ w.r.t cost function can be formulated in a similar way. However, the numerical error of this method is in the order of $\epsilon$. E.g., if $\epsilon = 0.1$, this method will yield an error in the order of 0.01. Why? Please close your eyes and think for a moment before moving on? 

Because:

$$
\begin{gather*}
f(\theta + \epsilon) = f(\theta) + \epsilon f'(\theta) + O(\epsilon^2)
\\
=>
\\
f'(\theta) = \frac{f(\theta + \epsilon) - f(\theta)}{\epsilon} = f'(\theta) + \frac{O(\epsilon^2)}{\epsilon} = O(\epsilon)
\end{gather*}
$$

One way to reduce this error is to do

$$
\begin{gather*}
f'(\theta) = \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon}
\end{gather*}
$$

To apply gradient checking on a single parameter $w_i$:

1. Apply foreprop, and backprop to get gradient of $w_i$, $g_i$.
2. Apply a small change to $w_i$, then do foreprop, backprop, and get gradient $g_i'$
3. Calculate:

$$
\begin{gather*}
\frac{||g_i - g_i'||}{||g_i|| + ||g_i'||}
\end{gather*}
$$

If the result is above $10^{-3}$, then we should worry about it.

## Things To Note In Gradient Checking

- **One thing to note is gradient check does NOT work with dropout. Because the final cost is influenced by turning off some other neurons as well.**
-  Let the neural net run for a while. Because when w and b are close to zero, the wrong gradients may not surface immediately