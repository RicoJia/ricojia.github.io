---
layout: post
title: Deep Learning - Integrated Gradients
date: 2026-10-22 13:19
subtitle: A Step-by-Step Example with a Tiny Neural Network
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Deep Learning
  - Machine-Learning
---

Integrated Gradients (IG) explains a neural network's prediction by estimating how much each input feature contributes to a score **relative to a chosen baseline input**. For an image classifier, it answers: which pixels favor cat, which favor dog, and by how much?

Ordinary gradients measure sensitivity at a single point. IG instead accumulates gradients along a path from a **baseline image** to the actual image.

## 1. Why Not Just Use Gradient × Activation?

Gradient × Activation multiplies each input by the gradient at the current input:

$$
G_i=x_i\frac{\partial S}{\partial x_i}
$$

where $$x_i$$ is an input feature (e.g. the red value of one pixel) and $$S$$ is the target score. The problem: **a network's gradient changes as the input changes.**

Take $$S(x)=\operatorname{ReLU}(x-2)$$ at $$x=4$$:

$$
S(4)=2,\qquad \left.\frac{dS}{dx}\right|_{x=4}=1,\qquad G(4)=4\cdot1=4
$$

But the actual change from zero is $$S(4)-S(0)=2$$. Gradient × Activation applies the gradient at $$x=4$$ to the whole interval $$[0,4]$$, yet the ReLU is inactive (gradient zero) for $$x<2$$. IG avoids this by accumulating the gradient along the interval.

---

## 2. The Integrated Gradients Equation

For a score $$S=f(x)$$ and **a baseline $$x'$$** (a black image, a blurred image, or another reference):

$$
\boxed{
IG_i(x)=(x_i-x_i')\int_0^1
\frac{\partial S\big(x'+\alpha(x-x')\big)}{\partial x_i}\,d\alpha
}
$$

- $$x$$, $$x'$$: actual and baseline inputs.
- $$\alpha\in[0,1]$$: interpolation factor; $$x'+\alpha(x-x')$$ is an intermediate input.
- $$IG_i$$: attribution of feature $$i$$ relative to the baseline.

With a zero baseline, the path is simply $$x(\alpha)=\alpha x$$: from all zeros at $$\alpha=0$$ to the original input at $$\alpha=1$$. We run the network at each point on the path, average the gradients, and multiply by the input difference:

$$
IG_i=(\text{input}-\text{baseline})\times\text{average gradient along the path}
$$

---

## 3. A Tiny Cat-Versus-Dog Network

```
RGB Image (3 × 2 × 2)
          |
          v
 Global Average Pooling
          |
          v
   RGB averages (3)
          |
          v
 Fully Connected (3 → 2)
          |
          v
        ReLU
          |
          v
       Softmax
          |
          v
   P(cat), P(dog)
```

### Input image

Four RGB pixels, written as one 2 × 2 matrix per channel (simplified values, not normalized RGB):

$$
R=\begin{bmatrix}2&1\\0&0\end{bmatrix},\quad
G=\begin{bmatrix}0&0\\0&1\end{bmatrix},\quad
B=\begin{bmatrix}0&0\\2&0\end{bmatrix}
$$

Global average pooling gives:

$$
\bar R=0.75,\qquad \bar G=0.25,\qquad \bar B=0.5
$$

### Classifier

$$
W=\begin{bmatrix}2&0&-1\\-1&0&2\end{bmatrix},\qquad
b=\begin{bmatrix}-0.5\\-0.1\end{bmatrix}
$$

The biases make the two ReLUs switch on at different points along the IG path. The pre-Softmax scores are:

$$
\begin{aligned}
z_{\text{cat}}&=\operatorname{ReLU}(2\bar R-\bar B-0.5)=\operatorname{ReLU}(1.5-0.5-0.5)=0.5\\
z_{\text{dog}}&=\operatorname{ReLU}(-\bar R+2\bar B-0.1)=\operatorname{ReLU}(-0.75+1-0.1)=0.15
\end{aligned}
$$

Softmax gives $$P(\text{cat})=\frac{e^{0.5}}{e^{0.5}+e^{0.15}}\approx0.587$$ and $$P(\text{dog})\approx0.413$$, so the model predicts cat.

### Attribution target

We explain the logit margin rather than the cat score alone or a probability (Softmax adds nonlinearity and couples the classes):

$$
\boxed{S=z_{\text{cat}}-z_{\text{dog}}}=0.5-0.15=0.35
$$

Positive $$S$$ favors cat; negative favors dog.

---

## 4. Integrated Gradients Step by Step

### Step 1: Choose the baseline

Use an all-zero image, $$x'=0$$. Then $$z_{\text{cat}}=\operatorname{ReLU}(-0.5)=0$$ and $$z_{\text{dog}}=\operatorname{ReLU}(-0.1)=0$$, so $$S(x')=0$$.

The goal is to distribute $$S(x)-S(x')=0.35$$ among the 12 input values.

### Step 2: Walk the path

Along $$x(\alpha)=\alpha x$$, pooling is linear, so $$\bar R=0.75\alpha$$, $$\bar G=0.25\alpha$$, $$\bar B=0.5\alpha$$, and:

$$
z_{\text{cat}}(\alpha)=\operatorname{ReLU}(\alpha-0.5),\qquad
z_{\text{dog}}(\alpha)=\operatorname{ReLU}(0.25\alpha-0.1)
$$

The dog unit switches on at $$\alpha=0.4$$ and the cat unit at $$\alpha=0.5$$, giving three gradient regions.

### Step 3: Gradients in each region

Each pixel contributes $$\frac14$$ to its channel average. Since $$S=z_{\text{cat}}-z_{\text{dog}}$$:

| Region | Active units | $$\partial S/\partial R_i$$ | $$\partial S/\partial B_i$$ |
|---|---|---|---|
| $$0\le\alpha<0.4$$ | none | $$0$$ | $$0$$ |
| $$0.4<\alpha<0.5$$ | dog | $$-\frac{-1}{4}=0.25$$ | $$-\frac{2}{4}=-0.5$$ |
| $$0.5<\alpha\le1$$ | cat and dog | $$\frac{2-(-1)}{4}=0.75$$ | $$\frac{-1-2}{4}=-0.75$$ |

Green gradients are always zero because the green weights are zero. The values exactly at the switching points do not affect the integral.

### Step 4: Integrate

$$
\begin{aligned}
\int_0^1\frac{\partial S}{\partial R_i}\,d\alpha&=0(0.4)+0.25(0.1)+0.75(0.5)=\boxed{0.4}\\
\int_0^1\frac{\partial S}{\partial B_i}\,d\alpha&=0(0.4)-0.5(0.1)-0.75(0.5)=\boxed{-0.425}
\end{aligned}
$$

### Step 5: Multiply by the input difference

With a zero baseline, $$IG_i=x_i\times(\text{integrated gradient})$$. Per channel:

$$
IG_R=0.4\begin{bmatrix}2&1\\0&0\end{bmatrix}=\begin{bmatrix}0.8&0.4\\0&0\end{bmatrix},\quad
IG_G=\begin{bmatrix}0&0\\0&0\end{bmatrix},\quad
IG_B=-0.425\begin{bmatrix}0&0\\2&0\end{bmatrix}=\begin{bmatrix}0&0\\-0.85&0\end{bmatrix}
$$

Summing over channels gives the attribution map:

$$
\boxed{
IG=\begin{bmatrix}+0.8&+0.4\\-0.85&0\end{bmatrix}
}
$$

| Pixel | IG | Meaning |
|---|---|---|
| Top-left | +0.80 | Favors cat over dog |
| Top-right | +0.40 | Favors cat over dog |
| Bottom-left | −0.85 | Favors dog over cat |
| Bottom-right | 0 | No attribution in this model |

These are contributions to the change in the logit margin relative to the baseline, not to Softmax probability.

---

## 5. Completeness

IG satisfies **completeness**: the attributions sum to the score difference from the baseline.

$$
\boxed{\sum_i IG_i=S(x)-S(x')}
$$

Check: $$0.8+0.4-0.85+0=0.35=S(x)-S(x')$$.

**Why it holds.** Let $$F(\alpha)=S\big(x'+\alpha(x-x')\big)$$. By the chain rule:

$$
F'(\alpha)=\sum_i\frac{\partial S}{\partial x_i}(x_i-x_i')
$$

Integrating from 0 to 1:

$$
\underbrace{F(1)-F(0)}_{S(x)-S(x')}
=\sum_i(x_i-x_i')\int_0^1\frac{\partial S}{\partial x_i}\,d\alpha
=\sum_i IG_i
$$

This holds for differentiable functions and for piecewise-differentiable networks such as ReLU networks, where the path derivative exists almost everywhere. In practice the integral is approximated numerically, so expect a small completeness error.

---

## 6. Compare with Gradient × Activation

At the original image ($$\alpha=1$$) both ReLUs are active, so $$\partial S/\partial R_i=0.75$$ and $$\partial S/\partial B_i=-0.75$$:

| Pixel | Gradient × Activation | Integrated Gradients |
|---|---|---|
| Top-left | +1.50 | +0.80 |
| Top-right | +0.75 | +0.40 |
| Bottom-left | −1.50 | −0.85 |
| Bottom-right | 0 | 0 |
| **Sum** | **+0.75** | **+0.35** |

Gradient × Activation sums to 0.75 because it uses only the gradients at $$\alpha=1$$. In that ReLU region the score is affine:

$$
S=\underbrace{3\bar R-3\bar B}_{0.75}-0.4=0.35
$$

Gradient × Activation recovers the feature term but misses the effective bias of $$-0.4$$. IG accounts for the gradient changing along the path and distributes the full score difference.

This does not make IG a more faithful causal explanation in general; it answers a more precisely defined, baseline-relative question.

---

## 7. Implementation in PyTorch

The integral is approximated with $$N$$ interpolation steps, using midpoint samples $$\alpha_k=\frac{k-1/2}{N}$$:

$$
\boxed{
IG_i\approx(x_i-x_i')\,
\frac1N\sum_{k=1}^{N}
\left.\frac{\partial S}{\partial x_i}\right|_{x=x'+\alpha_k(x-x')}
}
$$

```python
import torch

def integrated_gradients(model, image, baseline, steps=100):
    total_grad = torch.zeros_like(image)
    for k in range(steps):
        alpha = (k + 0.5) / steps
        x = (baseline + alpha * (image - baseline)).detach().requires_grad_(True)
        logits = model(x)  # pre-Softmax
        score = logits[:, 0] - logits[:, 1]  # cat-versus-dog margin
        total_grad += torch.autograd.grad(score.sum(), x)[0] / steps
    return (image - baseline) * total_grad
```

The attribution has the input's shape, `N × 3 × H × W`. Sum over channels for a spatial heatmap, keeping the sign when explaining a margin:

```python
heatmap = attribution.sum(dim=1)
```

Check completeness numerically:

```python
with torch.no_grad():
    z_x, z_b = model(image), model(baseline)
delta_s = (z_x[:, 0] - z_x[:, 1]) - (z_b[:, 0] - z_b[:, 1])
error = attribution.flatten(1).sum(dim=1) - delta_s  # → 0 as steps grow
```

---

## 8. Why the Baseline Matters

IG explains the difference between the input and a reference, so **the baseline defines what the attribution means**:

| Baseline          | Question answered                                          |
| ----------------- | ---------------------------------------------------------- |
| Black image       | What does this image contribute relative to black?         |
| Blurred image     | What does the detail contribute beyond a blurred version?  |
| Another dog image | What explains the score difference between the two images? |

A black image is easy to use but can be far from the training data, and the straight-line intermediate images may not look natural either.

Averaging IG over references sampled from a distribution (**Expected Gradients**) reduces dependence on any single baseline. It does not remove baseline bias; the explanation becomes relative to that reference distribution.

---

## 9. Integrated Gradients Versus Grad-CAM

| Property | Grad-CAM | Gradient × Activation | Integrated Gradients |
|---|---|---|---|
| Typical target | CNN feature maps | Input or intermediate activations | Input features |
| Spatial gradient averaging | Yes | No | No |
| Baseline | None | Implicit zero | Explicit |
| Evaluates inputs along a path | No | No | Yes |
| Completeness | Not generally | Not generally | Yes (exact integral) |
| Cost | One backward pass | One backward pass | Many backward passes |

- **Grad-CAM:** convenient for broad regions that influence a CNN classification.
- **Gradient × Activation:** position-specific local attribution in a feature map.
- **Integrated Gradients:** an exact score-difference decomposition over input pixels and channels, relative to a reference.

None of them proves that a highlighted feature caused the decision; perturbation experiments are an additional check.

---

## 10. Conclusion

IG attributes the score difference between an input and a baseline to individual input features by accumulating gradients along the path between them:

$$
\boxed{
IG_i=(x_i-x_i')\int_0^1\frac{\partial S\big(x'+\alpha(x-x')\big)}{\partial x_i}\,d\alpha,
\qquad
\sum_iIG_i=S(x)-S(x')
}
$$

In the cat-versus-dog example, the attributions $$+0.8$$, $$+0.4$$, $$-0.85$$ and $$0$$ sum to exactly the $$0.35$$ logit-margin difference from the zero baseline, while Gradient × Activation overshoots to $$0.75$$.

**Gradient × Activation uses the gradient at the current input; Integrated Gradients accumulates gradients from a reference to the input.** That is what lets it account for ReLU units switching on and off.
