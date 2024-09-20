---
layout: post
title: Deep Learning - RNN
date: '2022-03-04 13:19'
subtitle: Recurrent Neural Network!
comments: true
header-img: "img/home-bg-art.jpg"
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

> Is the world all about Transformers? No. RNN is small and beautiful. It's useful for SLAM in image debluring.

## Basics

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