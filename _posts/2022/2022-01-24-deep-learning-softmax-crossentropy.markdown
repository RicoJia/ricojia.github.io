---
layout: post
title: Deep Learning - Softmax And Cross Entropy Loss
date: '2022-01-24 13:19'
subtitle: Softmax, Cross Entropy Loss, and MLE 
comments: true
header-img: "img/home-bg-art.jpg"
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

Before introducing cross entropy as a loss, let's talk about entropy. Entropy is a measure of how "disperse" a system is. For a random variable, if its distribution is fairly "concentrated", its entropy is fairly small. E.g., a random variable with $P(x_1) = 0$ and $P(x_2) = 1$. This system is "concentrated" and its entropy is zero. On the other hand, if a random variable is "all over the place", like in a uniform distribution, then its entropy is high. The entropy is represented as:

$$
\begin{gather*}
H(P) = - \sum_x P(x)log(P(x))
\end{gather*}
$$

Similarly, we can define cross entropy as:

$$
\begin{gather*}
H(P,Q) = - \sum_x P(x)log(Q(x))
\end{gather*}
$$

- Cross entropy can be thought of as the "cost" to represent the distribution $P(x)$ using distribution $Q(x)$.

Now, to measure the "information loss" for using $Q(x)$ to represent $P(x)$, one metric is the **Kullback-Leibler (KL) Divergence**:

$$
\begin{gather*}
D_{KL}(P || Q) = - \sum_x P(x)log(\frac{P(x)}{Q(x)}) 
\\ = \sum_x P(x)log(Q(x)) - \sum_x P(x)log(P(x)) = H(P,Q) - H(P)
\end{gather*}
$$

When evaluating a model across epochs, **the true distribution and its entropy $H(P)$ remains the same for each input**. The difference of the model's performance each time is determined by $H(P,Q)$. Thus, the cross entropy can be used as a loss

$$
\begin{gather*}
\text{cross entropy loss J} = - \sum_x P(x)log(Q(x)) = - \sum_i y_i log(\hat{y_i})
\end{gather*}
$$

- Note that we always represent the ground truth label $y_i$ as an one-hot vector, where the true class has a probability 1, and other classes have a probability 0. So the one hot vector actually represents the true output distribution. **This is why cross entropy can be used as a loss in deep learning.**. $i$ is the dimension of the output vector.

**Cross-entropy loss** can be used with other activation functions like **ReLU**, **tanh**, etc., as long as we want the logit to be the probability distribution of the output.

## Softmax With Cross Entropy Loss

So combine Softmax and Cross Entropy Loss together:

1. Given output one-hot vector $\hat{y}$, compute softmax of each output class in one prediction

$$
softmax(\hat{y}) = p_i = \frac{1}{\sum_i e^{\hat{y_i}}}
\begin{bmatrix}
e^{\hat{y1}}, e^{\hat{y2}} ...
\end{bmatrix}
$$

2. Compute cross entropy against the target one-hot vector $y$. Because only 1 class has a probability 1, it can be simplified to:

$$
\begin{gather*}
L = -\sum y_i \cdot log(\hat{p_i}) = - log(\hat{p_m}) 
\\
= - \hat{y_m} + log(\sum_i(e^{\hat{y_i}}))
\end{gather*}
$$

Where `m` is the correct predicted class. This trick is also called the **log-sum-exp** trick

**The biggest advantage of softmax with cross-entropy-loss is numerical stability**. Some values in $e^{\hat{y}}$ can be large, so we subtract values by the largest element in $\hat{y}$, $\hat{y_{max}}$. So, we get $\sum_i e^{\hat{y_i} - \hat{y_{max}}}$ for better stability

In PyTorch, it is `torch.nn.CrossEntropyLoss()`, In TensorFlow, it is `tf.nn.softmax_cross_entropy_with_logits(labels, logits)`

## Gradient Descent With Softmax

The gradient of cross entropy loss, $J$ w.r.t jth dimension of the output prediction $\hat{y_j}$ is:

$$
\begin{gather*}
\frac{\partial J}{\partial y_j} = \frac{\partial \sum_i -y_i log(\hat{y_i})}{\partial \hat{y_i}} = \frac{\partial -y_jlog(\hat{y_j})}{\partial \hat{y_i}} = \frac{-y_j}{\hat{y_j}}
\end{gather*}
$$

The gradient of the Softmax layer's output at jth dimension $\hat{y_j}$, w.r.t jth dimension of the input $z_j$ is (that is, its own logit):

$$
\begin{gather*}
\hat{y}_j = \frac{e^{z_j}}{\sum_{K} e^{z_k}}
\\
\frac{\partial \hat{y}_j}{\partial z_j} = \frac{\partial}{\partial z_j} \left( \frac{e^{z_j}}{\sum_{k=1}^{C} e^{z_k}} \right)

\end{gather*}
$$

Using the quotient rule:

$$
\begin{gather*}
\frac{\partial \hat{y}_j}{\partial z_j} = \frac{\left(\sum_{k=1}^{C} e^{z_k}\right) \cdot \frac{\partial e^{z_j}}{\partial z_j} - e^{z_j} \cdot \frac{\partial}{\partial z_j} \left(\sum_{k=1}^{C} e^{z_k}\right)}{\left(\sum_{k=1}^{C} e^{z_k}\right)^2}
\end{gather*}

\\
= \frac{e^{z_j} \cdot \sum_{k=1}^{C} e^{z_k} - e^{z_j} \cdot e^{z_j}}{\left(\sum_{k=1}^{C} e^{z_k}\right)^2}

\\
= \frac{e^{z_j} \cdot \left(\sum_{k=1}^{C} e^{z_k} - e^{z_j}\right)}{\left(\sum_{k=1}^{C} e^{z_k}\right)^2}
$$

Since:

$$
\begin{gather*}
\hat{y}_j = \frac{e^{z_j}}{\sum_{k=1}^{C} e^{z_k}}
\end{gather*}
$$

We can get:

$$
\begin{gather*}
\frac{\partial \hat{y}_j}{\partial z_j} = \hat{y}_j \left(1 - \hat{y}_j\right)
\end{gather*}
$$


Similarly, the gradient of the Softmax layer's output at jth dimension $\hat{y_j}$, w.r.t cth dimension of the input $z_j$ is:


$$
\begin{gather*}
\frac{\partial \hat{y}_j}{\partial z_c} = \frac{\partial}{\partial z_c} \left( \frac{e^{z_j}}{\sum_{k=1}^{C} e^{z_k}} \right)
\\
= \frac{0 \cdot \sum_{k=1}^{C} e^{z_k} - e^{z_j} \cdot e^{z_c}}{\left(\sum_{k=1}^{C} e^{z_k}\right)^2}
\\
= -\frac{e^{z_j} \cdot e^{z_c}}{\left(\sum_{k=1}^{C} e^{z_k}\right)^2}
\\
= -\hat{y}_j \cdot \hat{y}_c
\end{gather*}
$$