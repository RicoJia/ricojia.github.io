---
layout: post
title: Deep Learning - Hyper Parameter Tuning 
date: '2022-01-22 13:19'
subtitle: It finally comes down to how much compute we have, actually...
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## How To Sample For Single Parameter Tuning

Generally, we need to try different sets of parameters to find the best performing one.

In terms of number layers, it could be a linear search:  

1. Define a range of possible numbers of layers e.g., $[5, 20]$
2. Uniformly sample from this range

However, the same cannot be applied to learning rate, momentum parameter, or RMS prop parameter. That's because they could range from $(0, 1]$. Therefore, the search would look like:

1. Choose a range of log10 of posible values. E.g., if we want $[1e-3, 1]$ for learning rate, we choose $[-3, 0]$
2. Uniformly sample in the log space.

For example, to estimate `alpha`:

```python
r = -5 * np.random().rand()
// gives 1e-5 to 1
alpha = 10 ** r
```

## Tuning Strategy On A High Level

If you have lots of compute, you might be able to train multiple models at a time. If not, you might want to spawn 1 set of model parameters, "babysit" the training process, and based on the result, decide what the next set of parameters could be. This process could take days.

Another point is that **hyperparameters could get stale**. As the business grows, you might want to re-evaluate the hyper parameters and retrain.

Importance ratings:
`learning_rate > num_hidden_layers > num_nodes_layer > Adam parameters`

When trying different values, trying random values is better than trying on a grid, because you might be trying different values.

Coarse to grid is the process to zoom in on a specific region in the hyper parameter space.
