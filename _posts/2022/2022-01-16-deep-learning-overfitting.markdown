---
layout: post
title: Deep Learning - Overfitting
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

### Technique 1: Direct Regularization

Regularization is to reduce overfitting by reduce the complexity of the model. Direct regularization does this by penalizing high weights of the model. This is also called **"weight decay"**. Common methods include:

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

Drop out is to force a fraction of neurons to zero during each iteration. This effectively reduces the complexity of the model, which is like regularization.

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
