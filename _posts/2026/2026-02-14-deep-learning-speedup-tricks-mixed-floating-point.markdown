---
layout: post
title: Deep Learning - Mixed Floating Point Training
date: '2026-02-14 13:19'
subtitle: FP16, BF16, Mixed Precision Training
comments: true
header-img: img/post-bg-kuaidi.jpg
tags:
    - Deep Learning
---

## Refresher: Floating Point Calculation

A floating point is represented as `sign bit | exponent | mantissa`. `0 | 10000001 | 10000000000000000000000` represents 6 because:

- Sign bit `0` represents positive.
- In IEEE 754, an FP32 number's exponent has a bias of 127. So the exponent `10000001` is `129-127=2`
- Mantissa (fraction) is 23-bit mantissa `10000000000000000000000`. In IEEE 754, there's an implicit leading 1 in the mantissa, so we interpret this as `1.10000000000000000000000` in binary, and `1.5` in decimal
  - `.10000000000000000000000` is 0.5, because the 1st digit is $2^{-1} = 0.5$, 2nd digit is $2^{-2} = 0.25$
- So all together, the value is:

$$
\begin{gather*}
\text{value} = (-1)^\text{exponent} \times 2^\text{exponent} \times \text{1.mantissa}
\\
= (-1)^\text{0} \times 2^\text{2} \times \text{1.5} = 6
\end{gather*}
$$

## BF16 vs FP16

[This section is inspired by this blogpost](https://medium.com/@furkangozukara/what-is-the-difference-between-fp16-and-bf16-here-a-good-explanation-for-you-d75ac7ec30fa) and [this blogpost](https://www.53ai.com/news/qianyanjishu/2024052494875.html)

- FP16 has `|1 sign bit | 5 exponent bits | 10 mantissa bits |`
  - Mantissa calculation for 1.5625: `1.1001000000 =  2^0 + 1 * 2^(-1) + 0 * 2^(-2) + 0 * 2^(-3) + 1 * 2^(-4) + 0 * 2^(-5) + 0 * 2^(-6) + 0 * 2^(-7) + 0 * 2^(-8) + 0 * 2^(-9) = 1.5625`

- BFloat16(Brain-Floating-Point-16) has `|1 sign bit | 8 exponent bits | 7 mantissa bits |`. This representation **sacrifices some precision for a wider range**. It was developed by Google Brain, and it's relatively new such that it's only supported on Nvidia Ampere+.

```
import transformers
transformers.utils.import_utils.is_torch_bf16_gpu_available()
```

BFloat16's dynamic range is `[-3.40282e+38，3.40282e+38]` whereas float16 is `[-65504，65504]`. Also, BF16 can go all the way down to ~10e-38 whereas FP16 has ~6e-8. So BFloat16 does NOT need scaling.

So, I'd suggest use BFloat16 when FP16 is suffering from exploding / vanishing gradient problem.

### Examples

- `0.0001`
  - FP16: `0|00001|1010001110`, which is 0.00010001659393.
        1. $0.0001 \approx 1.6384 \times 2^{−14}$
        2. Sign bit is 0 for positive.
        2. Actual Exponent `E_actual = -14`, so the FP16 exponent is `E = E_actual + bias = -14 + 15 = 1`. So we get `00001`
        3. For mantissa:
            1. FP16's mantissa **has an implicit leading 1**. So the mantissa represents `1.6384 - 1 = 0.6384`
            2. Convert `0.6384` to binary `1010001110`:

                ```
                0.6384 * 2 = 1.2768 -> 1
                0.2768 * 2 = 0.5536 -> 0
                0.5536 * 2 = 1.1072 -> 1
                0.1072 * 2 = 0.2144 -> 0
                0.2144 * 2 = 0.4288 -> 0
                0.4288 * 2 = 0.8576 -> 0
                0.8576 * 2 = 1.7152 -> 1
                0.7152 * 2 = 1.4304 -> 1
                0.4304 * 2 = 0.8608 -> 0
                0.8608 * 2 = 1.7216 -> 1
                ```

  - BF16: `0|01110001|1010010`, 0.00010013580322
        1. $0.0001 \approx 1.6384 \times 2^{−14}$
        2. Sign bit is 0 for positive.
        2. Actual Exponent `E_actual = -14`, so the FP16 exponent is `E = E_actual + bias = -14 + 127 = 113`. So we get `01110001`
        3. For mantissa: similar to the process for FP16.

### Precisions

- FP16 has 10 mantissa bits, which is `2^10=1024` numbers. So that's roughly `log_10(1024) = 3` significant digits.
- FP32 has 23 mantissa bits. So that is `log_10(2^23) = 7` significant digits.
- FP64 has 52 mantissa bits. So that's `log_10(2^52) = 15.6` significant digits

## Mixed Precision Training

[What Every User Should Know About Mixed Precision Training in PyTorch](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)

Matrix multiplcation, gradient calculation is faster if done in FP16, but results are stored in FP32 for numerical stability. So that's the need for mixed precision training.  Some ops, like linear layers and convolutions are faster in FP16. Other ops, like reductions, often require the dynamic range of float32

### Motivating Example - How FP16 Can Benefit Training

This is an example of linear regression

```python
import numpy as np

np.random.seed(42)
X = np.random.randn(100, 1)  # 100 samples, 1 feature

# Generate targets with some noise
true_W = np.array([[2.0]])
true_b = np.array([0.5])
Y = X @ true_W + true_b + 0.1 * np.random.randn(100, 1)

# Initialize weights and biases
W = np.random.randn(1, 1)  # Shape (1, 1)
b = np.random.randn(1)     # Shape (1,)

# Forward pass to compute predictions
def forward(X, W, b):
    return X @ W + b
def compute_loss(Y_pred, Y_true):
    return np.mean((Y_pred - Y_true) ** 2)

# Forward pass
Y_pred = forward(X, W, b)
loss = compute_loss(Y_pred, Y)

# Backward pass (compute gradients)
dLoss_dY_pred = 2 * (Y_pred - Y) / Y.size  # Shape (100, 1)

# Gradients w.r.t. W and b
dLoss_dW = X.T @ dLoss_dY_pred             # Shape (1, 1)
dLoss_db = np.sum(dLoss_dY_pred, axis=0)   # Shape (1,)

print("Gradients without scaling:")
print("dLoss_dW:", dLoss_dW)
print("dLoss_db:", dLoss_db)
```

- Without scaling, we see

```python
Gradients without scaling:
dLoss_dW: [[-1.9263151]]
dLoss_db: [-0.06291431]
```

- With scaling, **the main benefit is gradients are scaled up and avoid underflow using chain-rule**

```python
# forward pass, in loss
scaling_factor = 1024.0
scaled_loss = loss * scaling_factor
dScaledLoss_dY_pred = scaling_factor * dLoss_dY_pred

# backward()
# Scaled gradients w.r.t. W and b. THIS IS WHERE THE SCALING BENEFITS ARE FROM
dScaledLoss_dW = X.T @ dScaledLoss_dY_pred
dScaledLoss_db = np.sum(dScaledLoss_dY_pred, axis=0)

# scaler.step(optimizer), unscale the gradients, if there's no Nan or Inf
unscaled_dW = dScaledLoss_dW / scaling_factor
unscaled_db = dScaledLoss_db / scaling_factor

# update()
learning_rate = 0.1
W -= learning_rate * unscaled_dW
b -= learning_rate * unscaled_db
```
