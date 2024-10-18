---
layout: post
title: Deep Learning - RNN
date: '2022-03-04 13:19'
subtitle: Sequence Models
comments: true
header-img: "img/home-bg-art.jpg"
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Sequence Model

Some common sequence models include: DNA sequencing, audio clips, sentiment classification, etc. Another example is name indexing, where names in news for a past period of time will be searched so they can be indexed and searched appropriately. 

The first step to NLP is to build a dictionary, $X$. Say we have the most common 10000 English words, we can use one-hot encoding to represent a word, and the word can be indexed as $x^{i}$. If we see a word that's not in the dictionary, the word is indexed $x^{unk}$ as "unknown".

A fully connected network doesn't work well for sequence data: 1. The sequence could be of arbitrary lengths 2. There's no weight sharing in these sequence models.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/b8ad0eb3-25bf-4dea-9e6f-cc6d6a903e06" height="200" alt=""/>
    </figure>
</p>
</div>

## Basics

The word "recurrent" means "appearing repeatedly". In an RNN, we have What's it called??? $a$, sequential input $x$, output $\hat{y}$. Each superscript $i$ represents timestamp, e.g., $a^1$ means a at time $1$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/3e339dde-e16d-43ae-99c3-e689dc20cbcd" height="200" alt=""/>
    </figure>
</p>
</div>

$$
\begin{gather*}
a^{t} = g_0(W_{aa} a^{t-1} + W_{ax} x^{t} + b_a)
\\
\hat{y}^{i} = g_1(W_{ya} a^{t} + b_y)
\end{gather*}
$$

- $a^{0}$ is usually zero or randomly generated values.
- $g_0$ could be tanh (more common) or relu, $g_1$ could be sigmoid.
- $W_ax$ is a matrix that "generates a-like vectors, and takes in an x-like vector". Same notation for $W_aa$

We can simplify this notation into:

$$
\begin{gather*}
W_{a} = [W_{aa}, W_{ax}]
\\

a^{t} = g_0(W_{a}[a^{t-1}, x^{t}]^T + b_a)
\end{gather*}
$$


Disadvantage: inputs in the early sequence are not influenced by inputs later in the sequence. One example is "Teddy bear is on sale", Teddy is not a name, while many times it is.

## Old Notes (TODO)

FCN or CNN cannot handle sequential data well (time series data). Recurrent network has 1 input, 1 output , and output at each time is fed back into itself N times? (how does weight sharing work here?)

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/5d2a4106-60b5-46d0-9e77-b3176416ea33" height="200" alt=""/>
    </figure>
</p>
</div>

All RNN "layers" share the same weights: U, V, W. If we unfold inputs at different times:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/35add241-e857-4710-8cdb-3d5e057f3972" height="200" alt=""/>
    </figure>
</p>
</div>

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/fa68f6d1-dc25-40e3-a07e-ae820d8cd657" height="200" alt=""/>
    </figure>
</p>
</div>

- `s (hidden units) = tanh(W*output_{t-1} + U*x_{t})`
- `output = softmax(V * (W*output_{t-1} + U*x_{t}))`

Each training sample is a time series with vectors of the same dimensions, but its length can vary. Backpropagation Through Time is used here: **BPTT**

RNN can have exploding/diminising gradient as well. For explosion, do gradient clipping. for diminishing, can try 1. weight init, 2. use relu instead of sigmoid 3. other RNNs: LSTM, GRU.