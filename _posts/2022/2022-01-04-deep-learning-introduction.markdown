---
layout: post
title: Deep Learning - Introduction
date: '2022-02-04 13:19'
subtitle: Why Do We Even Need Deep Neuralnets?
comments: true
tags:
    - Deep Learning
---


## Why Do We Need Deep Learning

Any bounded continuous function can be approximated by an arbitrarily large single layer. W hy? The idea is roughly that the linear combinations of activation function can compose a pulses, like a sigmoid, or 4 ReLus. Then, the pulses can compose an arbitrary function

<p align="center">
<img src="https://github.com/RicoJia/Machine_Learning/assets/39393023/d1020b0c-776f-47c5-971f-b673d27e587b" height="300" width="width"/>
<figcaption align="center">Pulses Can Approximate An Arbitrary Function</figcaption>
</p>

## ML Design Process

```text
Choose dataset -> Choose network architecture (number of layers, number of hidden units on a layer, activation function at each layer, etc.)
```

Nowadays, It's very hard to guess all these params right in one shot. Intuitions from one domain (speech recognition) doesn't transfer well to another. **So going over the above iterations is key.**

Another secret sauce is data partition. There are:
- Training set
- Hold out - for choosing the best model hyperparameters (number of layers, learning rate, etc.) before testing on the test set
- Validation (or test) set
- traditionally, the ratio of Training vs Hold out vs Validation is: 70/20/10. For Big data: 98%/1%/1%

Come from the same train/test distribution.

### Bias And Variance

Overfitting = high variance, underfitting = high bias.

- Variance means **the difference from "high performance in training data, low performance in test data"**. That scenario is also called "overfitting".
- Bias means **the difference between human performance and training data performance**. Poor performance on training set is "underfitting". 
    - So if your human error is 15%, then the 15% ML model error rate is not considered high bias.
    - The same model could be high biased in certain landscapes (meaning humans can do well, but the model is underfitting even in the training set), and high variance in others (high performance in the training set, but low performance in the validation set)

When having high bias? Try a larger Network, or even a different architecture.
When having high Variance: more data, regularization, neural network architecture.