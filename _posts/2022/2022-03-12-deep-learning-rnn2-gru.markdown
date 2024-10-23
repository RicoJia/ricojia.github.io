---
layout: post
title: Deep Learning - RNN Part 2 GRU
date: '2022-03-12 13:19'
subtitle: Vanishing Gradients of RNN, GRU
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## The Vanishing Gradient Problem of RNN

RNN can doesn't handle long range dependencies well. One example is in speech recognition, "The cat which ate, slept, played and had a good day ... , was full" could be mispredicted as "the cat which ate, slept, played and had a good day ... , were full". This is because when we forward propagate, information of the previous part of the sequence is encoded in the hidden state $a^{(t)}$. As the input sequence gets longer, that information vanishes. (This is also called the "subject-verb agreement error")

In general, RNN does NOT have very good long range memory. To analyze why this problem occurs, we need to look at the training process. There, RNN suffers from the vanishing gradient problem as well. If we unroll the RNN with very long inputs, the network could be very deep. During Back-Propagation Through Time (BPTT), the gradient vanish / explode exponentially. In this case, during training, the gradient of the word `were` may not be updated with the gradient that penalizes plural nouns, if those nouns are too far from them so $a^{(t)}$ doesn't represent them much.

$$
\begin{gather*}
\frac{\partial L}{\partial W_ax^{(t)}} = \frac{\partial L}{\partial a^{(T)}} \cdots \frac{\partial a^{(t+1)}}{\partial a^{(t)}} a^{(t)}
\end{gather*}
$$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/c7a956fb-60cf-4de1-8953-db5e406e7573" height="300" alt=""/>
    </figure>
</p>
</div>

- And when doing back-prop, loss associated with `were` (which is wrong) could be too far from `cat`. So when updating gradients, the weights at `cat` doesn't get much loss from predicting `was` wrong. This leads to wrong association between `cat` and `was`
- Gradient explosion tends not to be an issue
- [HOMEWORK] Gradient clipping is: if supassing a threshold, **scale** the gradient

## Gated-Recurrent Unit (GRU)

To address the above issue, we increase "the memory" of the RNN.

1. We define **memory cell**, `C`. C is the same as input a $C^{(t)} = a^{(t)}$, but that's not the case in LSTM.
2. We introduce an intermediate value, $\tilde{C}$, 
3. In the meantime, calculate the gate coefficient $\Gamma_u$ from $C^{(t-1)}$, $x^{(t)}$ using sigmoid
    - $u$ means "update"

$$
\begin{gather*}
\tilde{C}^{(t)} = tanh(W_c [C^{(t-1)}, X^{(t)}] + b_c)
\\
\Gamma_u = \sigma(W_u[C^{(t-1)}, x^{(t)}] + b_u)
\\
C^{(t)} = \Gamma_u * \tilde{C}^{(t)} + (1 - \Gamma_u) * \tilde{C}^{(t-1)}
\end{gather*}
$$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/369951ac-19e0-484f-8ea0-db2d1b1db1bf" height="400" alt=""/>
    </figure>
</p>
</div>

## Example

Assume we have a vocabulary of 6: "[cat, which, ate, was, were, full]", with arbitrarily assigned values:

1. When processing the sound of "cat", its encoding $x^{1} = [3, 2, 1, 0, 0, 0]$

- Get intermediate output and gated value. Note that the dimension of $C$ and $x$ should be the same.

$$
\begin{gather*}
C^{(0)} = a^{(0)} = [0, 0, 0, 0, 0, 0]  \\
\tilde{C}^{(1)} = tanh(W_c [[0, 0, 0, 0, 0, 0], [3, 2, 1, 0, 0, 0]] + b_c) = [0.9, -0.9, -0.9, 0.9, 0.9, 0.9] \\
\Gamma_u = \sigma(W_u[[0, 0, 0, 0, 0, 0], [3, 2, 1, 0, 0, 0]] + b_u) = [0.8, 0, 0, 0, 0, 0] \\
\end{gather*}
$$

- This gives the final hidden state and output:

$$
\begin{gather*}
C^{(1)} = \Gamma_u * \tilde{C}^{(t)} + (1 - \Gamma_u) * \tilde{C}^{(t-1)} = [0.72, 0, 0, 0, 0, 0] \\
\hat{y}^{(1)} = softmax(W_{ya} [0.72, 0, 0, 0, 0, 0]  + b_y) = [0.8, 0.1, 0.1, 0, 0, 0]
\end{gather*}
$$

- $C^{[1]}[0]$ is the learned indication of "pluralism". Without the gate, the hidden state doesn't learn this pluralism, hence future outputs have no idea about this

2. When processing the sound for "which", we get $\tilde{C}$ and $\Gamma_u = [0.1, 0.1, 0.2, 0.1, 0.2, 0.4]$. It happens such that:

- $C^{(2)}[0]$ gets to largely retain $C^{[1]}$.
- Meanwhile, other elements in $C^{(2)}$ are different from those in $C^{[1]}$.
- The final output is $[0.1, 0.8, 0.1, 0, 0, 0]$, pointing to the word "which"

3. Similar situation in words before `"was"`. $C^{(t)}[0]$ gets to largely retain $C^{[t-1]}$ , but other elements gets different.

4. When processing "was", since $C^{(5)}[0] \approx 0.8$, the final output $[0.1, 0.1, 0.1, 0.31, 0.29, 0.1]$ shows "was" should be chosen instead of "were"

## Full GRU

$$
\begin{gather*}
\tilde{C}^{(t)} = tanh(W_c [\Gamma_r * C^{(t-1)}, x^{(t)}] + b_c)
\\
\Gamma_r = \sigma(W_r[C^{(t-1)}, x^{(t)}] + b_r)
\\
\Gamma_u = \sigma(W_u[C^{(t-1)}, x^{(t)}] + b_u)
\\
C^{(t)} = \Gamma_u * \tilde{C}^{(t)} + (1 - \Gamma_u) * \tilde{C}^{(t-1)}
\end{gather*}
$$

- `*` is elementwise multiplication
- $\Gamma_r$ coefficient is introduced to further weight $\tilde{C}^{(t-1)}$