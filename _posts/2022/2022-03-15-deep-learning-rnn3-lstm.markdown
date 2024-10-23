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
a^{\langle t \rangle} &= \Gamma_o * \tanh\left(c^{\langle t \rangle}\right)
\end{gather*}
$$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/11073e5f-cd14-4a8e-b4d6-c2eceadc76a5" height="300" alt=""/>
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
