---
layout: post
title: "[ML] Soft Rounding - Differentiable Quantization for Neural Networks"
date: 2026-08-08 13:19
subtitle:
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine Learning
---

Soft rounding is a smooth approximation to normal rounding. Normal rounding maps a continuous value to a discrete level:

$$
1.2 \rightarrow 1
$$

$$
1.8 \rightarrow 2.
$$

This is useful for modeling sensors that quantize range or intensity values. The problem is that ordinary rounding has zero gradient almost everywhere, so it is difficult to train a neural network through it. Soft rounding keeps the same staircase-like behavior, but replaces each hard jump with a smooth transition.

One possible definition is

$$
\lfloor x \rfloor
+
0.5
+
\frac{
\tanh(\tau r)
}{
2\tanh(\tau/2)
},
$$

where

$$
x-\lfloor x \rfloor-0.5.
$$

- The parameter $\tau$ controls how sharp the transition is.
- A small $\tau$ gives a smoother transition.
- A large $\tau$ makes the function behave more like normal rounding.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.sstatic.net/Bs1HQ.png" height="300" alt=""/>
    </figure>
</p>
</div>

## Small numerical example

Suppose

$$
x=1.3
$$

and

$$
\tau=2.
$$

First,

$$
\lfloor 1.3 \rfloor = 1.
$$

Then

$$
1.3-1-0.5 = -0.2.
$$

Substitute this into the soft-rounding equation:

$$
1
+
0.5
+
\frac{
\tanh(-0.4)
}{
2\tanh(1)
}.
$$

Using

$$
\tanh(-0.4)\approx -0.380
$$

and

$$
\tanh(1)\approx0.762,
$$

we obtain

$$
\operatorname{soft round}(1.3,2)
\approx
1.25.
$$

Hard rounding would give

$$
\operatorname{round}(1.3)=1.
$$

Soft rounding instead gives approximately

$$
1.25.
$$

The value is being pulled toward the quantized level, but the transition remains smooth.

## Why differentiability matters

With hard rounding,

$$
q*
\operatorname{round}
\left(
\frac{R}{q}
\right),
$$

small changes to $R$ usually do not change the output at all. For example,

$$
1.1,\ 1.2,\ 1.3,\ 1.4
$$

might all round to

$$
1.
$$

The gradient is therefore zero through most of the interval. Soft rounding replaces that flat jump with a smooth slope. A downstream loss can therefore tell the model whether increasing or decreasing $R$ would improve the result.  This is useful when the quantization itself is part of a learnable sensor model.

## The role of $q$ and $\tau$

The two parameters control different things. The quantization step $q$ controls the spacing of the levels. For example, if

$$
q=2\text{ mm},
$$

the approximate quantization levels are

$$
0,\ 2,\ 4,\ 6,\ldots\text{ mm}.
$$

The parameter $\tau$ controls how sharply values move between those levels. So:

$$
q
\rightarrow
\text{where the quantization levels are}
$$

while

$$
\tau
\rightarrow
\text{how hard or soft the transition is}.
$$

In a learnable sensor model, $q$ can be learned if the true quantization spacing is unknown. The sharpness $\tau$ is often fixed or gradually increased during training so that the model becomes closer to hard quantization over time.

Soft rounding therefore provides a practical compromise: it approximates the discrete behavior of a real sensor while still allowing gradients to pass through the quantization process.
