---
layout: post
title: Deep Learning - Hands-On Dinosour Name Generator Using RNN
date: '2022-03-21 13:19'
subtitle: 
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
    - Hands-on
---

## General Introduction

Build a character-level text generation model using an RNN.

- Data looks like:

```
{   0: '\n',
    1: 'a',
    2: 'b',
    3: 'c',
    4: 'd',
    5: 'e',
...
}
```

- At each step, output $y^{(t)}$ is fed back into the network as $x^{(t+1)}$

- Goal:
  - The goal is to train the RNN to predict the next letter in the name, so the labels are the list of characters that are one time-step ahead of the characters in the input `X`.
    - For example, `Y[0]` contains the same value as `X[1]`
    - So this is one example "dinarous", but spread out in multiple time steps?

  - The RNN should predict a newline at the last letter, so add `ix_newline` to the end of the labels.
    - Append the integer representation of the newline character to the end of `Y`.
    - Note that `append` is an in-place operation.
    - It might be easier for you to add two lists together.

## Workflow

Training Phase:

1. We feed a sequence of input characters X into the RNN.
2. The model outputs a sequence of probability distributions $y_{hat}$, each over the entire vocabulary.
3. The true labels $Y$ are the next characters in the sequence, shifted by one time step from X, so $X(1) = Y(0)$
4. We compute the cross-entropy loss between $y_{hat}$ and $y$
5. This loss is used to perform gradient descent and update the model's parameters, enabling it to learn to predict the next character in a sequence.

Inference Phase:

1. Given an initial character (or a special start token), the model generates the next character by predicting the most probable character that follows, by sampling from the probability distribution $y_{hat}$
2. This predicted character is then used as the input for the next time step.
3. The sequence generation continues iteratively, each time feeding the previously generated character back into the model.
4. The sequence terminates when the model predicts a newline character '\n', indicating the end of the word (dinosaur name).
