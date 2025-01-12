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

- The raw unnomalized output of the network, which is the input of the softmax layer, $Y_i$ is also know as **"Logit"**.

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
        <img src="https://github.com/user-attachments/assets/d86b297f-264c-41d0-90f8-2dd166a3fb90" height="200" alt=""/>
    </figure>
</p>
</div>

### What Softmax Really Does

What this classifier really does is to solve an Maximum Likelihood Estimation Problem (MLE). MLE is **given certain inputs, find a set of weights such that its predicted outputs have the highlest likelihood to match the observations.** So in this context:

- Softmax is usually used as the final layer. Its output represents the probability distribution of all output classes $\hat{Y}$ given certain inputs: $P(\hat{Y}|X)$.

- Is the MLE specific to Softmax? No.  MLE is the general theoretical framework of many types of models. However for classification, Softmax is a common and more intuitive way to implement MLE, especially when it's combined with **Cross-Entropy Loss**.

## Cross Entropy Loss (a.k.a Softmax Loss)

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

### Softmax With Cross Entropy Loss And Log-Sum-Exp Trick

By combining the softmax activation and cross-entropy loss into a single operation, implementations avoid computing the softmax probabilities explicitly, which enhances numerical stability and computational efficiency.

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

In PyTorch, it is `torch.nn.CrossEntropyLoss()`, In TensorFlow, it is `tf.nn.softmax_cross_entropy_with_logits(labels, logits)`. **This function requires masks to be in long (int64)**

## Cross Entropy Loss Variants

### Categorical Cross Entropy

Categorical Cross Entropy loss is designed for multi-class problems. There are two forms of the same function:

- Categorical Cross-Entropy: expects input labels to be **one-hot vectors**.
- Sparse Categorical Cross-Entropy: expects input labels to be **integers**. It is memory-efficient since it doesn't require one-hot encoding.

The purpose of using the Log-Sum-Exp Trick (LSE) is numerical stability. It helps prevent Overflow/Underflow where logits (raw inputs into softmax) can be very large or very small.

#### TensorFlow Implementations

- `tf.keras.losses.SparseCategoricalCrossentropy(from_logits: bool)`
  - If `from_logits=True`, tf will expect predictions being a probability distribution. Example:

        ```python
        y_true = [1, 2]
        y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
        scce = keras.losses.SparseCategoricalCrossentropy()
        scce(y_true, y_pred)
        ```

- Categorical cross-entropy loss functions will transfrom one-hot encoded vectors into integers, then utilize `SparseCategoricalCrossentropy`.

- `torch.nn.CrossEntropyLoss()` works with 1-hot labels. Internally, it first converts them to class indices, Then applies LSE (like categorical cross-entropy)

#### PyTorch Implementations

- `nn.CrossEntropyLoss(ignore_index)`
  - Input for per-pixel loss with `C` number of classes should have dimension: `(minibatch,C,d1​,d2​,...,dK​) with K≥1K≥1 for the K-dimensional case`
  - Input target will be in the shape of `(minibatch, d1, d2..dk)`. Each number should be the class label

### Binary-Cross-Entropy Loss

For binary classifiers, the output layer is usually 1 single unit. To normalize the output to `[0, 1]`, it's common to use the **sigmoid function** (NOT softmax):

$$
\begin{gather*}
\sigma(x) = \frac{1}{1 + e^{-x}}
\end{gather*}
$$

Then, calculate the loss:

$$
\begin{gather*}
L = -(ylog(\sigma(\hat{y})) + (1-y)log(1-\sigma(\hat{y})))
\end{gather*}
$$

In PyTorch, this is `torch.nn.BCEWithLogitsLoss()`. **Using this function is more numerically stable than calculating the sigmoid and Cross-Entropy separately.** Here is the reason:

- Above can be written as:

$$
\begin{gather*}
-[-log(1 + e^{-\hat{y}})y + (1-y)(log^{-\hat{y}} - log(1 + e^{-|\hat{y}|}))]
\\
=
\\
max(\hat{y}, 0) - \hat{y}y + log(1 + e^{-|\hat{y}|})
\end{gather*}
$$

When $\hat{y} \rightarrow \infty$, using the vanilla function will lead to underflow and you will get $-\infty$. The updated function is more accurate.

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
