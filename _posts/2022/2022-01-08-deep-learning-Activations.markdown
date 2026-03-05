---
layout: post
title: Deep Learning - Activation 
date: '2022-01-08 13:19'
subtitle: Sigmoid, ReLU, GELU Tanh
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Activation Functions

Early work observed that the Rectified Linear Unit (ReLU) often trains faster than sigmoid-like activations because it avoids saturation for positive inputs and has a simple gradient. Modern techniques such as batch normalization reduce some of the original differences, but ReLU and its variants remain popular.

- ReLU (`\mathrm{ReLU}(x)=\max(0,x)`):
  - Simple, computationally cheap, and has gradient 1 for positive inputs which helps gradient flow.
  - Advantages:
    - Avoids saturation on positive side; faster convergence in many networks.
    - Sparse activations (many zeros) can act as a regularizer and reduce computation.
  - Disadvantages:
    - "Dead" neurons: if a unit receives only negative inputs it can become inactive (output zero) and stop learning.
    - Unbounded outputs on the positive side.
  - Rule of thumb: avoid unnecessarily placing ReLU immediately before a softmax over logits; use a linear output for logits so relative differences are preserved.

    <p align="center">
        <img src="https://github.com/RicoJia/The-Dream-Robot/assets/39393023/d34a1631-c183-4a2e-b5f3-6bedc24b12a3" height="300"/>
    </p>

- Leaky ReLU: allows a small gradient for negative inputs, reducing dead neurons. For $\alpha>0$:

$$
\mathrm{LeakyReLU}(x)=\begin{cases}
x & x>0,\\
\alpha x & x\le 0.
\end{cases}
$$

    Typical choice: $\alpha\approx 0.01$.

    <p align="center">
        <figure>
                <img src="https://github.com/user-attachments/assets/0fc12aeb-8daf-4140-b09a-19d6e9b1fd5a" height="300" alt="Leaky ReLU"/>
        </figure>
    </p>

- ReLU6: a clipped ReLU that bounds the activation in `[0,6]`:

$$
\mathrm{ReLU6}(x)=\min(\max(0,x),6).
$$

    This is useful for quantized or mobile networks where a fixed activation range improves robustness to reduced numerical precision.

    <p align="center">
        <figure>
                <img src="https://github.com/user-attachments/assets/fd7ff666-de91-411b-82dd-b037b991370c" height="300" alt="ReLU6"/>
        </figure>
    </p>

- GELU (Gaussian Error Linear Unit): smoother alternative used in Transformers. Defined using the Gaussian CDF $\Phi(x)$; a common approximation is:

$$
\mathrm{GELU}(x)=x\,\Phi(x)\approx 0.5x\left[1+\tanh\left(\sqrt{\tfrac{2}{\pi}}\,(x+0.044715x^3)\right)\right].
$$

    Advantages: smooth, non-monotonic near zero, and avoids hard zeroing of negative inputs. Slightly slower to compute than ReLU.

    <p align="center">
        <figure>
                <img src="https://github.com/user-attachments/assets/8df49272-30cc-4335-9ac1-3cd02c9d37dd" height="300" alt="GELU"/>
        </figure>
    </p>

- tanh: a zero-centered sigmoid-like activation. Equivalent forms:

$$
 anh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}=\frac{2}{1+e^{-2x}}-1.
$$

    Range: $(-1,1)$. Compared with sigmoid, tanh is zero-centered which can help optimization, but it still saturates for large |x|.

    <p align="center">
        <figure>
                <img src="https://github.com/RicoJia/The-Dream-Robot/assets/39393023/22e4e9f7-8a9e-4e3c-9601-4f778281975c" height="300" alt="tanh"/>
        </figure>
    </p>

- Sigmoid (logistic):

$$
\sigma(x)=\frac{1}{1+e^{-x}}.
$$

    Range: $(0,1)$. Advantages: interpretable as a probability-like output; disadvantages: saturates for large |x| which leads to vanishing gradients (maximum derivative is $\sigma'(0)=0.25$).
