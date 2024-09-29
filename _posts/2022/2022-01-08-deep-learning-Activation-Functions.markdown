---
layout: post
title: Deep Learning - Activation and Cost Functions
date: '2022-01-08 13:19'
subtitle: Sigmoid, ReLU, Tanh, Mean Squared Error, Mean Absolute Error, Cross Entropy Loss, Hinge Loss, Huber Loss
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Activation Functions

Early papers found out that Rectified Linear Unit (ReLu) is always faster than Sigmoid because of its larger derivatives, and non-zero derivatives at positive regions. 

However, now with more tricks like batch normalization, 

1. ReLu: `y=max(0, x)`, at `x=0` it's technically non-differentiable. But we can still fake it simply: `y' = 0 or 1`  
    <p align="center">
    <img src="https://github.com/RicoJia/The-Dream-Robot/assets/39393023/d34a1631-c183-4a2e-b5f3-6bedc24b12a3" height="300" width="width"/>
    <figcaption align="center">ReLu</figcaption>
    </p>
    - Advantages:
        - This helps avoid diminishing derivatives? because large values have derivative being 1.
        - Computational efficiency. Easy to compute. 
        - It outputs zero on negative values, which makes forward pass compute less, as well as the backward pass
        - Because fewer neurons have non-zero outputs, the less "sensitive" the system is to smaller changes in inputs. Making it a form of regulation. (See the regularization section).
    - Disadvantages:
        - Exploding gradients: gradient can become large (because TODO?)
        - Dead neurons: neurons may not be activated if their weights make the output negative values.
        - Unbounded outputs

2. tanh: $\frac{2}{1+e^{-2x}}-1$
        <p align="center">
        <img src="https://github.com/RicoJia/The-Dream-Robot/assets/39393023/22e4e9f7-8a9e-4e3c-9601-4f778281975c" height="300" width="width"/>
        <figcaption align="center">tanh</figcaption>
        </p>
    - Advantages:
        - tanh starts from -1 to 1, so if we want negative values in output, go for tanh
        - compared to sigmoid, its derivative in `[-1, 1]` has a larger range, which could help with the learning process.
    - Disadvantages:
        - For large and small values, gradients are zero (vanishing gradient problem)
        - Could be moderately expensive to train (with exponentials)

3. Sigmoid (or "Logistic Function")
    <p align="center">
    <img src="https://github.com/RicoJia/The-Dream-Robot/assets/39393023/c552631a-861f-4c9b-a6b4-615130f53dab" height="300" width="width"/>
    <figcaption align="center">sigmoid</figcaption>
    </p>
    - Advantages:
    - Disadvantages:
        - For large and small values, gradients are zero (vanishing gradient problem)
            - Its max is 0.25. This could be a huge disadvantage given that nowadays with more layers, this gradient diminishes fast
        - Could be moderately expensive to train (with exponentials)

## Cost Functions

1. Mean Squared Error $\frac{1}{n} \sum(y_i-\hat{y}_i)^2$
    - Disadvantages:
        - Sensitive to outliers with larger errors due to the squaring (especially compared to MAE)
        - If errors are non-gaussian, this is probably not robust either.

2. Mean Absolute Error $\frac{1}{n} \sum(y_i-\hat{y}_i)$
    - Advantages:
        - Less sensitive to outliers
    - DisadvantagesL
        - Not differentiable at 0, which could be problematic, especially jacobians are near zero

3. Cross Entropy Loss (Log Loss): $-\frac{1}{n} \sum (y_ilog(\hat{y}_i) + (1-y^i)log(1-\hat{y}_i))$
    - Advantages:
        - Good for classification problems.
        - Suitable for probablistic Interpretation, and penalizes wrong confident predictions heavily
    - Disadvantages:
        - when $y^i$ is (0,1). And the closer it is to the bounds, the loss would be larger

4. Hinge Loss: $\frac{1}{n} \sum max(0, 1-y_i\hat{y}_i)$
    - Mostly used in SVM training, not working well with probablistic estimations

5. Huber Loss
    $$
    \frac{1}{2} (y_i - \hat{y}_i)^2, \text{for} |y_i - \hat{y}_i| < \delta
    \\
    \delta |y_i - \hat{y}_i| - \frac{1}{2} \delta^2, \text{otherwise}
    $$
    - Advantages:
        - it's continuous and differentiable
        - Combines MAE and MSE.
    - Requires tuning $\delta$