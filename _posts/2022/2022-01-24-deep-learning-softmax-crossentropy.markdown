---
layout: post
title: Deep Learning - Softmax And Cross Entropy Loss
date: '2022-01-24 13:19'
subtitle: Softmax, Cross Entropy Loss, and MLE 
comments: true
tags:
    - Deep Learning
---

## Softmax

When we build a classifier for cat classification, at the end of training, it's necessary to find the most likely classes for given inputs.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/e505c658-819c-42a7-87a9-15d9dc14c8bd" height="300" alt=""/>
    </figure>
</p>
</div>

- The raw unnomalized output of the network, which is the input of the softmax layer, $Y_i$ is also know as "Logit".

The softmax opertation is:

$$
\begin{gather*}
Z_i = \frac{e^{Y_i}}{\sum_i e^{Y_i}}
\end{gather*}
$$

Then, the classifier picks the highest scored class as its final output. 

### Properties of Softmax

- Softmax is effectively just choosing the highest scored predicted class, so it does not change how the raw score is calculated. So, if we have a perceptron network, all outputs will be a linear combination of the inputs. Adding a softmax will keep the decision boundaries linear 

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/d86b297f-264c-41d0-90f8-2dd166a3fb90" height="300" alt=""/>
    </figure>
</p>
</div>

### What Softmax Really Does

What this classifier really does is to solve an Maximum Likelihood Estimation Problem (MLE). MLE is **given certain inputs, find a set of weights such that its predicted outputs have the highlest likelihood to match the observations.** So in this context:

- Softmax is usually used as the final layer. Its output represents the probability distribution of all output classes $\hat{Y}$ given certain inputs: $P(\hat{Y}|X)$.

- Is the MLE specific to Softmax? No.  MLE is the general theoretical framework of many types of models. However for classification, Softmax is a common and more intuitive way to implement MLE, especially when it's combined with **Cross-Entropy Loss**.

## Cross Entropy Loss

## Gradient Descent With Softmax