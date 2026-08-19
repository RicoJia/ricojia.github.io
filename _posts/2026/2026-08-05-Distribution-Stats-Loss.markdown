---
layout: post
title: "[ML] Loss For Distribution Stats"
date: 2026-08-02 13:19
subtitle: loss for standard devations
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - robotics
---
# Matching Real Point-Cloud Statistics with a Differentiable Distribution Loss

Sometimes we want a synthetic point-cloud generator to produce range and intensity values that have the same statistical behavior as real sensor data.

For example, suppose we start with a clean synthetic range image and add learned sensor effects. We do not only want the output to look noisy. We want the generated residuals to have similar amplitude, distribution shape, spatial correlation, and local plateau behavior to residuals measured in real point clouds.

A useful way to do this is to compute a set of statistics from the generated data and compare them with the same statistics measured from real data.

The resulting loss is

$$  
L_{\text{dist}} = \sum_{k \in \text{scale}}  
w_k  
\left(  
\log s_k^{\text{gen}}

=\log s_k^{\text{real}}  
\right)^2  
+  
\sum_{k \in \text{shape}}  
w_k  
\left(  
s_k^{\text{gen}}

s_k^{\text{real}}  
\right)^2.  
$$

Here, $s_k^{\text{gen}}$ is a statistic measured from generated data, $s_k^{\text{real}}$ is the corresponding statistic measured from real data, and $w_k$ controls how important that statistic is.

The important question is: **what statistics should we put into this loss?**

## Start with the residual

Suppose the generator produces a range image $R_{\text{gen}}$. After removing the large-scale surface trend, we obtain a residual

$$  
x = R_{\text{gen}} - R_{\text{trend}}.  
$$

The residual represents the small-scale variation that we want to resemble real sensor measurements.

For a simple numerical example, imagine that one small region contains four residual values:

$$  
x =  
{1,\ 1,\ 1,\ 4}  
\text{ mm}.  
$$

Three pixels have a residual of $1$ mm, while one pixel has a larger $4$ mm residual. Different statistics describe different properties of this distribution.

## Overall noise amplitude: standard deviation

The first thing we may care about is simply how large the residual noise is. A convenient measure is the RMS value

$$  
L_2 =

\left(  
E|x|^2  
\right)^{1/2}.  
$$

For our example,

$$  
L_2 =

\sqrt{  
\frac{  
1^2 + 1^2 + 1^2 + 4^2  
}{4}  
}.  
$$
Therefore,

$$  
L_2 =

\sqrt{4.75}  
\approx  
2.18  
\text{ mm}.  
$$

For a zero-mean residual, this is closely related to the standard deviation. If the real residual has

$$  
\sigma_{\text{real}} = 2.0 \text{ mm}  
$$

but the generated residual has

$$  
\sigma_{\text{gen}} = 4.0 \text{ mm},  
$$

the generated noise amplitude is roughly twice as large as it should be. Because standard deviation is a positive scale quantity, it is useful to compare it in log space:

$$  
L_{\sigma} =

\left(  
\log \sigma_{\text{gen}}

\log \sigma_{\text{real}}  
\right)^2.  
$$

This makes the loss respond naturally to multiplicative errors.

## Distribution shape: $L_1/L_2$

Noise amplitude is not enough. Two residual distributions can have the same standard deviation while having very different shapes. To describe the central part of the distribution, we can use

$$  
L_1 =

E|x|.  
$$

For our example,

$$  
L_1 = \frac{  
1+1+1+4  
}{4}
=
1.75  
\text{ mm}.  
$$
We then divide by $L_2$:

$$  
\frac{L_1}{L_2} =

\frac{1.75}{2.18}  
\approx  
0.80.  
$$

Why divide by $L_2$? Because we want this statistic to describe **shape rather than absolute amplitude**. Suppose we multiply every residual by ten:

$$  
x =  
{10,\ 10,\ 10,\ 40}.  
$$

Both $L_1$ and $L_2$ become ten times larger, so their ratio stays the same:

$$  
\frac{L_1}{L_2}  
\approx  
0.80.  
$$

That means $L_1/L_2$ tells us about the structure of the distribution without being dominated by its overall scale. It is useful for describing the core of the residual distribution.

## Tail shape: $L_4/L_2$

To pay more attention to large residuals, we increase the exponent. Define

$$  
L_4 =

\left(  
E|x|^4  
\right)^{1/4}.  
$$

For our example,

$$  
L_4 =

\left(  
\frac{  
1^4+1^4+1^4+4^4  
}{4}  
\right)^{1/4}.  
$$

Since

$$  
4^4 = 256,  
$$

we obtain

$$  
L_4 =

\left(  
\frac{259}{4}  
\right)^{1/4}  
\approx  
2.84  
\text{ mm}.  
$$

Therefore,

$$  
\frac{L_4}{L_2} =

\frac{2.84}{2.18}  
\approx  
1.30.  
$$

Notice what happened. The large residual of $4$ mm contributes much more strongly to $L_4$ than to $L_1$ or $L_2$. So $L_4/L_2$ gives us a differentiable measure of how heavy the tails are.

---

## Why not directly use p99 or p999?

A natural alternative would be to compare percentiles such as p99 or p999. Those are excellent diagnostic measurements, but they are poor training losses. A percentile is obtained by sorting the residual values and selecting one particular element. For example, if there are $16{,}384$ valid pixels, a hard p999 calculation may send gradient through essentially one selected residual value. That is only about

$$  
\frac{1}{16384}  
\times 100  
\approx  
0.006%.  
$$

The selected pixel can also change abruptly when residual values reorder. By contrast,

$$  
L_p =

\left(  
E|x|^p  
\right)^{1/p}  
$$
depends on all of the residual values. The gradient is still weighted toward large residuals for large $p$, but the objective is much smoother than a hard quantile. This gives a useful division of labor:

$$  
\text{training}  
\rightarrow  
L_1/L_2,\ L_4/L_2,\ L_6/L_2  
$$

while

$$  
\text{evaluation}  
\rightarrow  
\text{MAD/std},\ \text{p99/std},\ \text{p999/std}.  
$$

The quantiles remain useful because they directly describe the real distribution. They simply do not need to be the quantities through which we backpropagate. That tells the optimizer that the deepest tail of the generated residual distribution differs substantially from the real one.

Because these statistics are differentiable, the gradient can travel backward through the entire pipeline:

$$  
\theta  
\rightarrow  
R_{\text{gen}}  
\rightarrow  
x  
\rightarrow  
\frac{L_1}{L_2},  
\frac{L_4}{L_2},  
\frac{L_6}{L_2}  
\rightarrow  
L_{\text{dist}},  
$$

where $\theta$ represents the learnable parameters of the generator.

Training therefore adjusts $\theta$ so that the generated statistics move toward the real statistics.

---

## Spatial structure also matters

Matching the histogram of residual values is not enough. Real sensor noise often has spatial structure. Neighboring range values may be correlated. Noise may differ along rows and columns. Quantization may create small local plateaus. We can therefore add several spatial statistics to the same loss.

For example, using auto-correlation-function:

$$  
\text{ACF}_{\text{row},1},  
\text{ACF}_{\text{row},2},  
\text{ACF}_{\text{row},3}  
$$

and

$$  
\text{ACF}_{\text{col},1},  
\text{ACF}_{\text{col},2},  
\text{ACF}_{\text{col},3}.  
$$

These measure how strongly residuals are correlated at different spatial offsets. We can also measure how frequently neighboring range values are nearly equal. A smooth version is

$$  
P_{\text{eq}}

= E_{\text{adjacent pairs}}  
\left[  
\exp  
\left(

\left(  
\frac{\Delta R}{\epsilon}  
\right)^2  
\right)  
\right],  
$$

where $\Delta R$ is the range difference between adjacent valid pixels and $\epsilon$ controls what counts as approximately equal.

For

$$  
\epsilon = 1 \text{ mm},  
$$

an adjacent difference of zero gives

$$  
\exp(0)=1.  
$$

A difference of $1$ mm gives

$$  
\exp(-1)  
\approx  
0.368.  
$$

A difference of $2$ mm gives

$$  
\exp(-4)  
\approx  
0.018.  
$$

So $P_{\text{eq}}$ becomes large when many neighboring range measurements are nearly identical. This is useful for matching plateau-like structure created by sensor quantization or discretization.

## Mean adjacent difference

Another useful statistic is

$$  
E|\Delta R_{\text{adjacent}}|.  
$$

This measures the typical amount of local frame-to-frame spatial variation. If generated range values change too aggressively between neighboring pixels, this statistic becomes too large. If the synthetic output is unnaturally smooth, it becomes too small. Because it is a positive scale quantity, it can be compared in log space.

## Putting everything together

A practical version of the statistics loss is therefore

$$  
L_{\text{stats}} =

L_{\text{scale}}  
+  
L_{\text{shape}}.  
$$

The scale part can contain

$$  
\sigma  
$$

and

$$  
E|\Delta R_{\text{adjacent}}|.  
$$

These are compared in log space:

$$  
L_{\text{scale}} = \sum_{k \in \text{scale}}  
w_k  
\left(  
\log s_k^{\text{gen}}

\log s_k^{\text{real}}  
\right)^2.  
$$

The shape and spatial part can contain

$$  
\frac{L_1}{L_2},  
\quad  
\frac{L_4}{L_2},  
\quad  
\frac{L_6}{L_2},  
$$

$$  
\text{ACF}_{\text{row},1..3},  
\quad  
\text{ACF}_{\text{col},1..3},  
$$

and

$$  
P_{\text{eq}}.  
$$

These can be compared directly:

$$  
L_{\text{shape}}

= \sum_{k \in \text{shape}}  
w_k  
\left(  
s_k^{\text{gen}}

s_k^{\text{real}}  
\right)^2.  
$$

The complete loss is

 $$  
\boxed{  
L_{\text{dist}}

= \sum_{k \in \text{scale}}  
w_k  
\left(  
\log s_k^{\text{gen}}

= \log s_k^{\text{real}}  
\right)^2  
+  
\sum_{k \in \text{shape}}  
w_k  
\left(  
s_k^{\text{gen}}

s_k^{\text{real}}  
\right)^2  
}  
$$

The purpose of this loss is not to force every generated point cloud to copy one particular real frame.

Instead, it gives the generator a set of differentiable statistical targets:

- how large the range noise should be,

- what the center of the residual distribution should look like,

- how heavy the tails should be,

- how extreme residuals should behave,

- how neighboring measurements should correlate,

- and how frequently local range plateaus should occur.
