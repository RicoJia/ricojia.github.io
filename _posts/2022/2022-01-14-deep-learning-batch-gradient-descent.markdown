---
layout: post
title: Deep Learning - Batch Gradient Descent
date: '2022-01-14 13:19'
subtitle: Batch Gradient Descent
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## A Neuron And Batch Gradient Descent

A Neuron, has multiple inputs and a single output. First it gets the weighted sum of all inputs, then feeds it into an "activation function". Below, the activation function $\sigma(z)$ is a "sigma function"

$$
\begin{gather*}
z = \sum_{i}^{n} w_ix_i = w_0x_0 + w_1 x_1 ...
\\
y = \sigma(z) = \frac{1}{1 + e^{-z}}
\end{gather*}
$$


<p align="center">
<img src="https://github.com/RicoJia/The-Dream-Robot/assets/39393023/49e6c48c-65e3-44c5-b130-db4138440499" height="300" width="width"/>
<figcaption align="center">Image Source: Stackoverflow</figcaption>
</p>

So for a single input, x (`nx1` vector), and its corresponding groudtruth value y, we can get its prediction $\hat{y}$ and cost $J$. In our example, J is cross-entropy, which has its minimum cost when $y = \hat{y}$

$$
\begin{gather*}
\hat{y} = \sigma(\sum_{i}^{n} w_ix_i)
\\
J = -(ylog(\hat{y}) + (1-y)log(1-\hat{y}))
\end{gather*}
$$

Then, we can apply gradient descent to update parameters, $w$, which is the **goal of deep learning**. The gradient is represented as $\nabla{J}$, which is the partial derivative w.r.t all parameters in vector $w$. Since gradient gives the steepest **ascent**, we want to update $w$ with the negative of that.

$$
\begin{gather*}
\nabla{J} = [\frac{\partial{J}}{\partial{w_0}}, \frac{\partial{J}}{\partial{w_1}} ...]
\\
w = w - \lambda\nabla{J}
\end{gather*}
$$

For an individual partial derivative value,

$$
\begin{gather*}
\frac{\partial{J}}{\partial{w_i}} = \frac{\partial{J}}{\partial{\hat{y}}} \frac{\partial{\hat{y}}}{\partial{z}} \frac{\partial{z}}{\partial{w_i}}
\\ 
\frac{\partial{J}}{\partial{\hat{y}}} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}
\\
\frac{\partial{\hat{y}}}{\partial{z}} = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} (1-\frac{1}{1 + e^{-z}}) 
= \hat{y}(1-\hat{y})
\\
\frac{\partial{z}}{\partial{w_i}} = x_i
\\
=>
\\
\frac{\partial{J}}{\partial{w_i}} = (\hat{y} - y)x_i
\\
\frac{\partial{J}}{\partial{w_i}} = (\hat{y} - y)
\\
\end{gather*}
$$

So from the single input $x$, we update $w$ with

$$
\begin{gather*}
w = w - \lambda\nabla{J} = w - \lambda(\hat{y} - y)x
\\
b = b - \lambda\nabla{J} = w - \lambda(\hat{y} - y)
\end{gather*}
$$

Ideally, we feed the entire batch of training set into the network and optimize our weights using gradient descent. This is called "vectorization", because we put the entire batch in a matrix. Matrix operations are usually faster than their for-loop counterparts, because they make use of low-level optimization techniques like SIMD, parallel computing, etc. 

**Now, if we have a batch inputs, that is $x^{(0)} ... x^{(m)}$**

$$
\begin{gather*}

J = -\frac{1}{m} \sum_{m}^{M} (y^{(m)} log(\hat{y}^{(m)}) + (1-y^{(m)})log(1-\hat{y}^{(m)}))
\\
w = w - \sum_{m}^{M}\lambda\nabla{J^{(m)}} = w - \frac{1}{m} \sum_{m}^{M} \lambda(\hat{y}^{(m)} - y^{(m)})x^{(m)}
\\
b = b - \lambda\nabla{J} = b - \lambda(\hat{y} - y)
\end{gather*}
$$

## Neural Network And Back Propagation

Now, to be consistent with mainstream notations, we represent the output of each node as $a$, instead of $\hat{y}$. 

Imagine we have two layers, an input layer, and an output layer. To update all params $w$ to yield better final output, we first pass inputs $x^{(0)} ... x^{(m)}$ through the network. The output of each node $Lj$ is $a^{L}_{j}$, where $L$ is the layer number. The target values are $y^{(0)} ... y^{(m)}$.

### When L is An Output Layer

$$
\begin{gather*}
J = -\frac{1}{m} \sum_q^Q(ylog(a^L_q) + (1-y)log(1-a^L_q))
\\
\frac{\partial{J}}{\partial{a^L_q}} = -\frac{1}{m} \frac{y-a^L_q}{a^L_q(1-a^L_q)}
\end{gather*}
$$

### When L is A Hidden Layer

For node $a^{Lj}$, we assume all nodes in layer $L-1$ are connected to $A^{Lj}$ (Fully Connected), outputs from $L-1$ is written as vector $a^{L-1}$. Note that $w^{L}_{j}$ is parameter vector and $z^{L}_{j}$ are scalar intermediate outputs

$$
\begin{gather*}
w = w - \sum_{m}^{M}\lambda\nabla{J^{(m)}} 
\\
\nabla{J^{(m)}} = \frac{\partial{J}}{\partial{w^L_j}} = \frac{\partial{J^L_j}}{\partial{a^L_j}} \cdot \frac{\partial{a^L_j}}{\partial{z^L_j}} \cdot \frac{\partial{z^L_j}}{\partial{w^L_j}}
\end{gather*}
$$

For a specific node in layer L, $Lj$:

$$
\begin{gather*}
\frac{\partial{z^{L}_{j}}}{\partial{w^{L}_{j}}} = [a^{L-1}_0, a^{L-1}_1 ... ] = a^{L-1}
\\
\frac{\partial{a^{L}}}{\partial{z^{L}}} = \dot{\sigma{(z^{L})}}
\end{gather*}
$$

The node is connected to nodes 0...Q in layer L+1, that is, partial derivative needs to consider all these influenced paths. With scalar output $a^{L}_{j}$, scalar derivative $\frac{\partial{J}}{\partial{a^{L}_{j}}}$

$$
\begin{gather*}
\\
\frac{\partial{J}}{\partial{a^{L}_{j}}} = \sum_{q}^{Q} \frac{\partial{J}}{\partial{a^{L+1}_q}} \frac{\partial{a^{L+1}_q}}{\partial{z^{L+1}_q}}
\frac{\partial{z^{L+1}_q}}{\partial{a^{L}_j}}
\end{gather*}
$$

To further reduce $\frac{\partial{J}}{\partial{a^{L}_j}}$ , with scalar derivative $\dot{\sigma{(z^{L+1}_q)}}$, node $(L+1,q)$'s parameter vector $w^{L+1}_{q}$, and the jth parameter $w^{L+1}_{q, j}$

$$
\begin{gather*}
z^{L+1}_q = w^{L+1}_q\cdot a^{L} => \frac{\partial{z^{L+1}}}{\partial{a^{L}_j}} = w^{L+1}_{q, j}
\\
a^{L+1} = \sigma{(z^{L+1})} => \dot{\sigma{(z^{L+1})}}
\\
=> \frac{\partial{J}}{\partial{a^{L}_j}} = \sum_q^Q\frac{\partial{J}}{\partial{a^{L+1}_q}} w^{L+1}_{q,j} \dot{\sigma{(z^{L+1})}}
\end{gather*}
$$

## Summary 
That was a lot of details with chain rule. So in all, during back propagation, assume we have batch size of m, input size n and output size p.
1. Start from the output layer and back track 
    - For context , compute $\frac{\partial{J}}{\partial{a^{L}_j}}$, given J being the binary cross entropy

        $$
        \begin{gather*}
        \frac{\partial{J}}{\partial{a^L_q}} = \frac{y-a^L_q}{a^L_q(1-a^L_q)} \text{,if L is the output layer}
        \end{gather*}
        $$

    - Then, for all nodes in the layer, and for all batches:

        $$
        \begin{gather*}
        \frac{\partial{J}}{\partial{a}^L} = \frac{y-a}{a(1-a)}
        \\
        \frac{\partial{a}}{\partial{z}^L} = \dot{\sigma{(z^{L})}}
        \\
        \frac{\partial{J}}{\partial{z^{L}}} = \frac{\partial{J}}{\partial{a}^L} \frac{\partial{a}}{\partial{z}^L}
        \end{gather*}
        $$
        - $y, a, z$ are `mxp` matrices.

2. For non-output layers,
    - For context, each individual neuron has

        $$
        \begin{gather*}
        \frac{\partial{J}}{\partial{a^{L}_j}} = \sum_q^Q\frac{\partial{J}}{\partial{z^{L+1}_q}} \frac{\partial{z^{L+1}_q}}{\partial{a^{L}_q}}
        = \sum_q^Q\frac{\partial{J}}{\partial{z^{L+1}_q}} w^{L+1}_{q,j}  \text{,if L is not an output layer}

        \\
        \frac{\partial{J}}{\partial{z^{L}_j}} = \frac{\partial{J}}{\partial{a^{L}_j}} \frac{\partial{a^{L}_q}}{\partial{z^{L}_q}}
        = \frac{\partial{J}}{\partial{a^{L}_j}} \dot{\sigma{(z^{L})}}
        \end{gather*}
        $$

    - Then, for all nodes in the layer, and for all batches:

        $$
        \begin{gather*}
        \frac{\partial{J}}{\partial{a^{L}}} = (W^{L+1})^T \frac{\partial{J}}{\partial{z^{L}}}
        \\
        \frac{\partial{J}}{\partial{z^{L}}} = \frac{\partial{J}}{\partial{a^{L}}} \frac{\partial{a^{L}}}{\partial{z^{L}}}
        = \frac{\partial{J}}{\partial{a^{L}_j}} \dot{\sigma{(z^{L})}}
        \end{gather*}
        $$


3. Finally, during the update step:
    - For context, each individual neuron has

        $$
        \begin{gather*}
        \frac{\partial{J}}{\partial{w^L_j}} = \frac{\partial{J^L_j}}{\partial{a^L_j}} \cdot \frac{\partial{a^L_j}}{\partial{z^L_j}} \cdot \frac{\partial{z^L_j}}{\partial{w^L_j}}
        \\
        w^L_j = w^L_j - \frac{1}{m}\sum_{m}^{M}\lambda \frac{\partial{J}}{\partial{w^L_j}}
        \\
        b^L_j = b^L_j - \frac{\partial{J}}{\partial{z^L_j}}
        \end{gather*}
        $$

    - Then, for all nodes in the layer, and for all batches:

        $$
        \begin{gather*}
        \frac{\partial{J}}{\partial{w^L}} = \frac{\partial{J}}{\partial{z^{L}}}  \frac{\partial{z^{L}}}{\partial{w^L}} = \frac{\partial{J}}{\partial{z^{L}}} a^{(L-1)}
        \\
        w^L = w^L - \frac{1}{m}\sum_{m}^{M}\lambda \frac{\partial{J}}{\partial{w^L}}
        \\
        b^L_j = b^L - \frac{\partial{J}}{\partial{z^L}}
        \end{gather*}
        $$

## Batch Gradient Descent, MiniBatch Gradient Descent, Stochastic Gradient Descent

Batch Gradient Descent is the original way of optimziation. It runs the entire batch of inputs, gets its total cost and sum of weight gradients across the entire batch, then optimizes the weights with that. However it takes too long when `m >2000`. 

SGD is fast, but the cost-epoch trajectory also can be noisy. 

Mini batches is the way to go. A typical mini-batch size is `64 - 512`

A comparison of these three batching methods of epoch costs is shown below: 

![Screenshot from 2024-08-11 15-01-49](https://github.com/user-attachments/assets/5c5fed43-623b-4dbf-afc0-a79eea1a43a5)
![Screenshot from 2024-08-11 15-01-40](https://github.com/user-attachments/assets/369d6d6e-79b7-455f-9a2d-3b428884c883)

### Mini-Batch Method

1. Shuffle the inputs and labels. Before feeding to the network, shuffle $X$ and the corresponding $Y$.

```python
permutation = list(np.random.permutation(m))
# each column is a feature
shuffled_X = X[:, permutation]
shuffled_Y = Y[:, permutation].reshape((1,m))
inc = i
```

2. Partition the input data into batches of size `m`. **Powers of 2 are often chosen to be batch sizes**