---
layout: post
title: "[ML] Straight Through Bernulli Output"
date: 2026-07-31 13:19
subtitle:
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - robotics
---

## 1. Goal

We want to use a CycleGAN to model a 3D imaging sonar that comes with random dropouts. The sonar has two channels: intensity and range. The dropout value is (0,0). As part of the CycleGAN network, we have a refiner subnetwork that models the correction to a raw simulated input. For each sonar beam, the refiner decides whether to keep or drop it. The intended behavior is:

```text
gate = 1 -> keep R and I
gate = 0 -> R = 0, I = 0
```

A naive implementation is:

```python
gate = gate * valid_in #valid_in is probablity

R_out = gate * R_refined
I_out = gate * I_refined
valid_out = gate.bool()
```

The CNN predicts a logit:

```text
H_P
```

A sigmoid converts it to a keep probability:

```text
p_keep = sigmoid(H_P)
```

Example:

```text
H_P = 1.4
p_keep ≈ 0.80
```

A Bernoulli sample converts the probability into `0` or `1`:

```text
gate ~ Bernoulli(p_keep)
```

So the beam has about:

```text
80% chance to survive
20% chance to drop
```

The problem is **that this hard decision is not differentiable**.

---

## 2. Mitigation Trick: Straight-through estimator

First compute a soft value:

```python
soft = sigmoid((H_P + noise) / tau)
hard = (soft > 0.5).float()
gate = hard + soft - soft.detach()
```

- `soft.detach()` has the same numerical value as `soft`, but its gradient is blocked.

This gives:

```text
forward pass  -> gate behaves like hard
backward pass -> gradient behaves like soft
```

### Example 1: beam survives

```text
soft = 0.80
hard = 1
```

Then:

```text
gate = 1 + 0.80 - 0.80
     = 1
# given:
R_refined = 13.2 m
I_refined = 25

=> 

R_out = 13.2 m
I_out = 25
```

During backpropagation:

TODO: why approximation? how does pytorch handle (p>0.5).float()?

```text
d gate / d H_P
≈ d soft / d H_P
= soft * (1 - soft) #For a sigmoid
= 0.8 * 0.2
= 0.16
```

So the dropout head still receives a gradient.

### Example 2: beam drops

Suppose:

```text
soft = 0.20
hard = 0
```

Then:

```text
gate = 0 + 0.20 - 0.20
     = 0

=> 

R_out = 0
I_out = 0

d gate / d H_P
≈ 0.2 * 0.8
= 0.16
```

So even a dropped beam can affect learning.

---

## 3. Randomness

To make dropout stochastic, add random noise before the sigmoid:

```python
U = Uniform(0, 1)
noise = log(U) - log(1 - U)

soft = sigmoid((H_P + noise) / tau)
hard = (soft > 0.5).float()

gate = hard + soft - soft.detach()
```

Different random samples give different dropout patterns for the same input.

---

## 4. CycleGAN limitation

If a beam is dropped:

```text
before:
R = 13.2
I = 25

after:
R = 0
I = 0
```

the original values are lost.

The reverse CycleGAN generator cannot know whether the original beam was. Therefore dropout is not invertible. Do not apply range or intensity cycle loss to dropped beams.

Use:

```text
cycle_mask =
    valid_before_forward
    AND
    valid_after_forward
```

and compute cycle loss only on surviving beams.
