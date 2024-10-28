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
    </figure>
</p>
</div>

3 layers is already pretty deep for RNN, especially if we look at the impact of inputs from the first few time stamps.
It's not uncommon to connect the output `y` to fully connected layers, so temporally, the deeper network is not connected.

## Back Propagation of LSTM

$$
\begin{gather*}
tanh'(x) = (1-tanh^2(x))
\\
\sigma'(x) = \sigma(x) (1 - \sigma(x))
\end{gather*}
$$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/ed4fd0c3-1398-4cbb-8cd0-09e9b6ee018c" height="300" alt=""/>
    </figure>
</p>
</div>

$$
\begin{gather*}
\begin{align}
d\gamma_o^{\langle t \rangle} &= da_{next}*\tanh(c_{next}) * \Gamma_o^{\langle t \rangle}*\left(1-\Gamma_o^{\langle t \rangle}\right)\tag{7} \\[8pt]
dp\widetilde{c}^{\langle t \rangle} &= \left(dc_{next}*\Gamma_u^{\langle t \rangle}+ \Gamma_o^{\langle t \rangle}* (1-\tanh^2(c_{next})) * \Gamma_u^{\langle t \rangle} * da_{next} \right) * \left(1-\left(\widetilde c^{\langle t \rangle}\right)^2\right) \tag{8} \\[8pt]
d\gamma_u^{\langle t \rangle} &= \left(dc_{next}*\widetilde{c}^{\langle t \rangle} + \Gamma_o^{\langle t \rangle}* (1-\tanh^2(c_{next})) * \widetilde{c}^{\langle t \rangle} * da_{next}\right)*\Gamma_u^{\langle t \rangle}*\left(1-\Gamma_u^{\langle t \rangle}\right)\tag{9} \\[8pt]
d\gamma_f^{\langle t \rangle} &= \left(dc_{next}* c_{prev} + \Gamma_o^{\langle t \rangle} * (1-\tanh^2(c_{next})) * c_{prev} * da_{next}\right)*\Gamma_f^{\langle t \rangle}*\left(1-\Gamma_f^{\langle t \rangle}\right)\tag{10}
\end{align}
\end{gather*}
$$

### Parameter Derivatives =

$$
\begin{gather*}
dW_f = d\gamma_f^{\langle t \rangle} \begin{bmatrix} a_{prev} \\ x_t\end{bmatrix}^T \tag{11}
\\
dW_u = d\gamma_u^{\langle t \rangle} \begin{bmatrix} a_{prev} \\ x_t\end{bmatrix}^T \tag{12}
\\
dW_c = dp\widetilde c^{\langle t \rangle} \begin{bmatrix} a_{prev} \\ x_t\end{bmatrix}^T \tag{13}
\end{gather*}
\\
dW_o = d\gamma_o^{\langle t \rangle} \begin{bmatrix} a_{prev} \\ x_t\end{bmatrix}^T \tag{14}
$$

To calculate $db_f, db_u, db_c, db_o$ you just need to sum across all 'm' examples (axis= 1) on $d\gamma_f^{\langle t \rangle}, d\gamma_u^{\langle t \rangle}, dp\widetilde c^{\langle t \rangle}, d\gamma_o^{\langle t \rangle}$ respectively. Note that you should have the `keepdims = True` option.

$$
\begin{gather*}
\displaystyle db_f = \sum_{batch}d\gamma_f^{\langle t \rangle}\tag{15}
\\
\displaystyle db_u = \sum_{batch}d\gamma_u^{\langle t \rangle}\tag{16}
\\
\displaystyle db_c = \sum_{batch}dp\widetilde c^{\langle t \rangle}\tag{17}
\\
\displaystyle db_o = \sum_{batch}d\gamma_o^{\langle t \rangle}\tag{18}
\end{gather*}
$$

Finally, you will compute the derivative with respect to the previous hidden state, previous memory state, and input.

$$
\begin{gather*}
 da_{prev} = W_f^T d\gamma_f^{\langle t \rangle} + W_u^T   d\gamma_u^{\langle t \rangle}+ W_c^T dp\widetilde c^{\langle t \rangle} + W_o^T d\gamma_o^{\langle t \rangle} \tag{19}
\end{gather*}
$$

Here, to account for concatenation, the weights for equations 19 are the first n_a, (i.e. $W_f = W_f[:,:n_a]$ etc...)

$$
\begin{gather*}
 dc_{prev} = dc_{next}*\Gamma_f^{\langle t \rangle} + \Gamma_o^{\langle t \rangle} * (1- \tanh^2(c_{next}))*\Gamma_f^{\langle t \rangle}*da_{next} \tag{20}

 \\
dx^{\langle t \rangle} = W_f^T d\gamma_f^{\langle t \rangle} + W_u^T  d\gamma_u^{\langle t \rangle}+ W_c^T dp\widetilde c^{\langle t \rangle} + W_o^T d\gamma_o^{\langle t \rangle}\tag{21}
\end{gather*}
$$

where the weights for equation 21 are from n_a to the end, (i.e. $W_f = W_f[:,n_a:]$ etc...)
