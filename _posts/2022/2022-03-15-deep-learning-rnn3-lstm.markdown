---
layout: post
title: Deep Learning - RNN Part 3 LSTM, Bi-Directional RNN, Deep RNN
date: '2022-03-15 13:19'
subtitle: 
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## LSTM

LSTM came out in 1997 and GRU is a simplification of it. In LSTM, we have the "forget gate", $\Gamma_r$, the output gate $\Gamma_o$, and the update gate $\Gamma_u$. We do NOT have $\Gamma_r$

$C^{(t-1)}$ can retain largely the $C^{(t)}$.

$$
\begin{gather*}
\tilde{c}^{\langle t \rangle} &= \tanh\left(W_c \left[a^{\langle t-1 \rangle}, x^{\langle t \rangle} \right] + b_c \right) \\
\Gamma_u &= \sigma\left(W_u \left[a^{\langle t-1 \rangle}, x^{\langle t \rangle} \right] + b_u \right) \\
\Gamma_f &= \sigma\left(W_f \left[a^{\langle t-1 \rangle}, x^{\langle t \rangle} \right] + b_f \right) \\
\Gamma_o &= \sigma\left(W_o \left[a^{\langle t-1 \rangle}, x^{\langle t \rangle} \right] + b_o \right) \\
c^{\langle t \rangle} &= \Gamma_u * \tilde{c}^{\langle t \rangle} + \Gamma_f * c^{\langle t-1 \rangle} \\
a^{\langle t \rangle} &= \Gamma_o * \tanh\left(c^{\langle t \rangle}\right) \\
y^{\langle t \rangle} &= \text{softmax}\left(W_{ya} * a^{\langle t \rangle} + b_y\right)
\end{gather*}
$$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/e7d9bd70-189b-4f43-a912-f6861e788228" height="300" alt=""/>
    </figure>
</p>
</div>

One variation is the "Peephole" connection. That is, the hidden state is introduced in deciding the gate values.

$$
\begin{gather*}
\Gamma_u &= \sigma\left(W_u \left[a^{\langle t-1 \rangle}, x^{\langle t \rangle}, C^{(t-1)} \right] + b_u \right) \\
\Gamma_f &= \sigma\left(W_f \left[a^{\langle t-1 \rangle}, x^{\langle t \rangle}, C^{(t-1)} \right] + b_f \right) \\
\Gamma_o &= \sigma\left(W_o \left[a^{\langle t-1 \rangle}, x^{\langle t \rangle}, C^{(t-1)} \right] + b_o \right) \\
\end{gather*}
$$

LSTM is historically the most proven architecture. GRU is simpler, newer, and potentially better for growing big

Detailed Explanations:

- $x^{(t)}, a^{(t-1)}$ are stacked vertically
- $\Gamma_f * C^{(t-1)}$ is like applying a mask.
- All gates should be in the ranges of $[0,1]$. That is, if close to 0, values from the previous state, or calculated intermediate state won't be kept
- So when a subject changes its state, like a singular noun changes to plural,  $\Gamma_f$'s certain values should change its value `0 -> 1`
- $\tilde{C}$ uses `tanh` so its values are in $[-1, 1]$. Whether its values are passed onto the actual hidden state $C^{(t-1)}$is determined by gate $\Gamma_i$
- $a = \Gamma_o * tanh(C^{t})$ to normalize its values to $[-1, 1]$

So LSTM is like RNN, but it also has a cell state $C$ (long term memory), $a$ (short term memory)

- A forget gate is added, so long term memory could be abandonded.
  - So the long term memory is the direct contribution of LSTM to address RNNs' vanishing gradient problem
- An update gate to make sure some current cell state candidate **might not** get into the short term memory
- An output gate determines which elements from the cell state gets into the short term memory.
  - Changes more rapidly (like a "scratch pad"), and goes directly into the next output.

## Bi-Directional RNN

Bi-Directional RNN can learn not only the correspondence from the past, but also from the "future". This is an acyclic graph.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/8927337c-3909-4dbf-8c0f-b3bb99225609" height="300" alt=""/>
        <figcaption><a href="">Source: </a></figcaption>
    </figure>
</p>
</div>

This is actually a good thing to try at the beginning

## Deep RNN

With RNN, GRU, LSTM blocks one can build deep RNNs.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/211e113a-f1c6-43e5-b329-996d46d86435" height="300" alt=""/>
        <figcaption><a href="">Source: </a></figcaption>
    </figure>
</p>
</div>

3 layers is already pretty deep for RNN, especially if we look at the impact of inputs from the first few time stamps.
It's not uncommon to connect the output `y` to fully connected layers, so temporally, the deeper network is not connected.
