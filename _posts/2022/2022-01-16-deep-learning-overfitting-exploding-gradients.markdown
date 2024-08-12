---
layout: post
title: Deep Learning - Overfitting, Exploding And Vanishing Gradients
date: '2022-01-16 13:19'
subtitle: When in doubt, be courageous, try things out, and see what happens! - James Dellinger
comments: true
tags:
    - Deep Learning
---


## A Nice Quote 💡

Before we delve in, I'd like to quote [from James Dellinger](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79) that really hits home: 

> I think the journey we took here showed us that this knee-jerk response of feeling of intimidated, while wholly understandable, is by no means unavoidable. Although the Kaiming and (especially) the Xavier papers do contain their fair share of math, we saw firsthand how experiments, empirical observation, and some straightforward common sense were enough to help derive the core set of principals underpinning what is currently the most widely-used weight initialization scheme.

I share the same fear when I see some scary-looking math, too. However, one lesson I learned from life is most things could be conquered, or resolved partially to a satisfactory state, by simply spending time and "drilling" into it, step by step. At the end of the day, many scary-looking things would shatter and wouldn't be so scary anymore.

### Bias And Variance

Overfitting = high variance, underfitting = high bias.

- Variance means **the difference from "high performance in training data, low performance in test data"**. That scenario is also called "overfitting".
- Bias means **the difference between human performance and training data performance**. Poor performance on training set is "underfitting", and that could lead to high bias
    - So if your human error is 15%, then the 15% ML model error rate is not considered high bias.
    - The same model could be high biased in certain landscapes (meaning humans can do well, but the model is underfitting even in the training set), and high variance in others (high performance in the training set, but low performance in the validation set)

- When having high bias? (underfitting) Try a larger Network, or even a different architecture.
    - Do not add regularization. WHy??? you might be suffering exploding gradients?


- When having high Variance, (overfitting): more data, regularization, neural network architecture. See the next section
    - **Do NOT use a bigger neural net.** This is because overfitting usually means too complex of a network structure. Making it bigger or deeper will add to the complexity of it.
    - So we need to decrease the complexity here. 

## Overfitting

Capacity is the ability to fit a wide variety of functions. Models with complex patterns may also be overfitting, thus have smaller capacity.

### Technique 1: Regularization

Regularization is to reduce overfitting by penalizing the "complexity" of the model. This is also called "weight decay". Common methods include:

- L1 and L2 regularization:
    - L1 encourages sparsity: **NOT SUPER COMMON** $\lambda \sum_j || w_j ||$
    - L2 penalizes large weights: $ \frac{\lambda}{2m} \sum_j || w_j^2 ||$. $b$ could be omitted. $\lambda$ is another parameter to tune (regularization parameter). $m$ is the output dimensions.
    - The regularization term is a.k.a "weight decay"

Effectively, some neurons' weight will be reduced, so hopefully, it will result in a simpler model that could perform better on the test set landscape.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/d832260c-662a-41aa-8140-4b4c99b77753" height="300" alt=""/>
        <figcaption>Source: Andrew Ng's Deep Learning Class on Coursera</figcaption>
    </figure>
</p>
</div>

### Technique 2: Dropout

Drop out is to force a fraction of neurons to zero during each iteration. Redundant representation

- At each epoch, **randomly select** neurons to turn on. Say you want 80% of the neurons to be kept. This means we will not rely on certain features. Instead, we shuffle that focus, which spreads the weights
- **VERY IMPORTANT**: **computer vision uses this a lot. Because you have a lot of pixels, relying on every single pixel could be overfitting.**

- So, there are fewer neurons effectively in the model, hence makes the decision boundary simpler and more linear.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/e9404d42-64ee-40f7-a949-e140db824006" height="300" alt=""/>
    </figure>
</p>
</div>

#### Notes For Dropout

- In computer vision, due to the nature of the data, it's default practice to apply drop out. **However in general, do not apply drop out if there's no overfitting.**

- One detail is that when calculating drop out, we need to scale the output of each layer by `1/keep_prob` so the final expected cost is the same. Accordingly, due to chain rule, the gradients of weights should be multiplied by `1/keep_prob` as well.

- To implement drop out, we can simply apply a "drop-out" mask in foreprop (multiplied by `1/keep_prob` ). When calculating the weight gradients, apply the same mask again. 

**During Inferencing, do NOT turn on drop-out**. The reason being, it will add random noise to the final result. You can choose to run your solution multiple times with drop out, but it's not efficient, and the result will be similar to that without drop-out.

But be careful with visualization of $J$, it becomes wonky because of the added randomness.

### Technique 3: Tanh Activation Function

Note: when the activation is tanh, when w is small, the intermediate output z of the neuron is small. tanh is linear near zero. So, the output model is more like a perceptron network, which learns linear decision boundaries. Hence, the model's decision boundary is likely to be more linear. Usually, overfitting would happen when the decision boudary is non linear.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/63808479-fa3c-4cbb-a5c4-7691587e5e06" height="300" alt=""/>
        <figcaption>Source: Andrew Ng's Deep Learning Class on Coursera</figcaption>
    </figure>
</p>
</div>

### Technique 4: Data Augmentation

One can get a reflection of a cat, add random distortions, rotations,

### Technique 5: Early Stopping

Stop training as you validate on the dev set. If you know realize that your training error is coming back up, use the best one.

**One thing to notice is "orthogonalization"**: the tools for optimizing J and overfitting should be separate.

## Exploding & Vanishing Gradients

In a very deep network, output of each layer might diminish, because these outputs are products $W_1W_2...x$ (ignoring activation for now)

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/e9d8659e-a778-4be2-90c0-88de1f620cbf" height="100" alt=""/>
    </figure>
</p>
</div>

Let's pretend that the above is a very deep network ;), and each node has exactly the same weights. Then, the final output is on the order of $W^nx$. So if $W$'s elements are slightly over 1, outputs could explode. On the other hand, if $W$'s elements are slightly below 1, output could diminish to 0. In [the batch gradient derivation](./2022-01-14-deep-learning-optimizations.markdown), we saw how gradient at one layer depends on the gradient of its output through **the chain rule**: $\frac{\partial{J}}{\partial{w_i}}$. Since the output is exponential w.r.t any layer, this gradient is also "exponential", intuitively.

$$
\begin{gather*}
\frac{\partial{J}}{\partial{w_i}} = \frac{\partial{J}}{\partial{\hat{y}}} \frac{\partial{\hat{y}}}{\partial{z}} \frac{\partial{z}}{\partial{w_i}}
\end{gather*}
$$

### Weight Initialization

Below is a summary from James' Dilenger's article [Weight Initialization in Neural Networks: A Journey From the Basics to Kaiming](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79)

Let's say we have a simple neural net like the one above. Each layer has 256 neurons, and the input is of size 256. We have 100 layers. Naively, without activation functions, if we initialize each layer randomly so that each weight has `mean=0`, `standard deviation=1`, it would look like:

```python
import torch
x = torch.randn(256)
w = torch.randn(256, 256)
for i in range(100):
    x = w@x
    if torch.isnan(x.norm()): break
i   #see 31
```

Assume we have normalized our inputs to `mean=0`, unit `standard deviation=1`. Then

In the above example, the kernel would have trouble even at layer 31! Why?!

#### Why Gradient Explodes

It's because for :

1. The output vector $y$ at layer 1 has 0 mean $0$, $1$.
2. The standard deviation at each layer will keep growing.

To see what's happening in this layer:

```python
import torch
mean, var = 0, 0
for i in range(10000):
    x = torch.randn(256)
    w = torch.randn(256, 256)
    y = w @ x
    mean += y.mean().item()
    var += y.pow(2).mean().item()   # this is E[y^2] = var(y)
mean/10000, var/10000
```

The math is that

$$
\begin{gather*}
y = Wx =
\begin{bmatrix}
w_{1, 1} & ... & w_{1, 256} \\
... \\
w_{256,1} & ... & w_{256, 256}
\end{bmatrix}

\begin{bmatrix}
x_1 \\
... \\
x_{256}
\end{bmatrix}

=

\begin{bmatrix}
y_1 \\
... \\
y_{256}
\end{bmatrix}
\end{gather*}
$$

In code, we can see that the mean is 0. This is because for independent variables $w$ and $x$: 

$$
\begin{gather*}
E[wx] = E[w]E[x] = 0
\end{gather*}
$$

And the standard deviation is almost $\sqrt{256}$. This is because for a single product $wx$: 

$$
\begin{gather*}
Var(wx) = E[(wx)^2] - E[(wx)] = E[(wx)^2] - 0 = E[w^2x^2] = E[w^2]E[x^2] = 1
\end{gather*}
$$

And for $y_i = w_{i1}x_{i} + ... + w_{i, 256}x_{i}$, its mean is 0. If we talk about $y_i$'s variance, it would be 256, because:

$$
\begin{gather*}
Var(A + B) = E[(A+B)^2] = E[A^2] + 2E[A^B] + E[B^2] = E[A^2] + E[A^2] = 2
\end{gather*}
$$

Then, the variance of $Y$ across all $y_i$ is 256. The standard deviation of $y$ is 16. If y is a Gaussian distribution, this means 33.3% of y will lie outside of $[-16, 16]$. 

In the next layer, the standard deviation will be amplified further and further. 

#### Naive Initialization For Perceptron Networks

What about we make each layer's weights so that each layer's output vector $y$ has `mean=0`, `standard deviation = 1`? This way, the next layer's $y$ will still be 1.

It's quite simple to do that. Since:

$$
\begin{gather*}
Var(y) = Var(y_i) = w_{i1}x_{i} + ... + w_{i, 256}x_{i}
\end{gather*}
$$

If we scale $W$'s variance by $\sqrt{1/256}$, problem solved, right? It seems so!

```python
import torch
x = torch.randn(256)
w = torch.randn(256, 256)/16
for i in range(100):
    x = w@x
    if torch.isnan(x.norm()): break
i   # see 99
```

#### Xavier Initialization

However, we have to add the non-linear activation back in so a neural net can classify complex patterns non-linearly in its landscape, such as handwritten digit classification. 

In Xavier Glorot and Yoshua Bengio's paper: [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf), they believed one good way to initialize is to:

1. Create uniformly distributed weights
2. Normalized them to $\frac{\sqrt{6}}{\sqrt{n_i + n_{i+1}}}$, where $n_i$ and $n_{i+1}$ are "fan in" and "fan out" (number of inputs and outputs to the layer)

```python
import math
import torch
def xavier(fan_in, fan_out):
    weight = torch.empty(fan_in, fan_out)
    w = torch.nn.init.uniform_(weight, -math.sqrt(6./(fan_in + fan_out)), math.sqrt(6./(fan_in + fan_out)))
    return w
x = torch.randn(256)
for i in range(100):
    w = xavier(256, 256)
    x = torch.tanh(w@x)
x.mean(), x.std()   # see (tensor(0.0047), tensor(0.0452))
```

#### He (Kaiming) Initialization

What if the activation function is **ReLu**? In his paper: [Delving Deep into Rectifiers:
Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852) (ICCV, 2015)  He Kaiming (何恺明) found that instead of scaling $\sqrt{1/n}$ (as in the naive scaling approach), we do $\sqrt{2/n}$, we could achieve good results. Why? Intuitively, half of the $y$ would turn to 0 after ReLu. So $y$'s variance will become half

#### Remarks On Zero Initialization

If our network's weights are initialized all to 0, then the output is the same value regardless of inputs. Gradients will be zero everywhere. This is called "symmetry". To break the symmetry, it's okay to initialize W zero and not including b. 

Also, if the output is 0 and uses the **cross entropy loss**, we will get `inf` in the final cost.

$$
\begin{gather*}
Loss = -y log(\hat{y}) + (1-y) log((1-\hat{y}))
\end{gather*}
$$