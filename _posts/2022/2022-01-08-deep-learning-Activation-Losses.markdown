---
layout: post
title: Deep Learning - Activation and Loss Functions
date: '2022-01-08 13:19'
subtitle: Sigmoid, ReLU, GELU Tanh, Mean Squared Error, Mean Absolute Error, Cross Entropy Loss, Hinge Loss, Huber Loss, IoU Loss, Dice Loss, Focal Loss
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Activation Functions

Early papers found out that Rectified Linear Unit (ReLu) is always faster than Sigmoid because of its larger derivatives, and non-zero derivatives at positive regions.

However, now with more tricks like batch normalization,

- ReLu: `y=max(0, x)`, at `x=0` it's technically non-differentiable. But we can still fake it simply: `y' = 0 or 1`  
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
    - Rule Of Thumbs of Applications:
        - ReLu should **NOT** be applied immediately before softmax, because it could distort the relative differences between logits by setting the negative ones to 0.

- Leaky ReLU: regular ReLU would lead to dead neurons when $x=0$. That could cause learning to be stuck. Leaky ReLU can unstuck this situation by having a small negative slope and allowing backprop flow correspondingly.

$$
\begin{gather*}
y = x (x >0 )
\\
y = \alpha x (x <=0)
\end{gather*}
$$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/0fc12aeb-8daf-4140-b09a-19d6e9b1fd5a" height="300" alt=""/>
    </figure>
</p>
</div>

- `ReLu6`
  - `ReLu 6 = min(max(x, 0), 6)`. So if there is an overflow, or underflow, outputs could be naturally large, and would be capped just as they would without overflow/underflow.
  - Activation values now occupy a smaller and fixed range, so in terms of quantization noise, so the representation is with fewer bits.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/fd7ff666-de91-411b-82dd-b037b991370c" height="300" alt=""/>
    </figure>
</p>
</div>

- Typically $\alpha=0.01$

- GELU (Gaussian Error Linear Units)
  - ReLu is less common in Transformer due to its dying negative activations. GELU is `x * Cumulative Density Function (CDF) of Gaussian Distribution at point x`.
  - Advantages:
    - It's differentiable all over $\R$. That helps with smoothening hard gradient transitions (like in ReLU)
    - Also, it has no dead neurons
  - Disadvantages:
    - It's slower

    $$
    \begin{gather*}
    \begin{aligned}
    & GELU(x) = 0.5x*[1 + tanh(\sqrt{\frac{2}{\pi}} (x + 0.044715 x^3))]
    \end{aligned}
    \end{gather*}
    $$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/8df49272-30cc-4335-9ac1-3cd02c9d37dd" height="300" alt=""/>
    </figure>
</p>
</div>

- tanh: $\frac{2}{1+e^{-2x}}-1$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/RicoJia/The-Dream-Robot/assets/39393023/22e4e9f7-8a9e-4e3c-9601-4f778281975c" height="300" alt=""/>
    </figure>
</p>
</div>

- Advantages:
  - tanh starts from -1 to 1, so if we want negative values in output, go for tanh
  - compared to sigmoid, its derivative in `[-1, 1]` has a larger range, which could help with the learning process.
- Disadvantages:
  - For large and small values, gradients are zero (vanishing gradient problem)
  - Could be moderately expensive to train (with exponentials)

- Sigmoid (or "Logistic Function")
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

3. Hinge Loss: $\frac{1}{n} \sum max(0, 1-y_i\hat{y}_i)$
    - Mostly used in SVM training, not working well with probablistic estimations

4. Huber Loss
    $$
    \frac{1}{2} (y_i - \hat{y}_i)^2, \text{for} |y_i - \hat{y}_i| < \delta
    \\
    \delta |y_i - \hat{y}_i| - \frac{1}{2} \delta^2, \text{otherwise}
    $$
    - Advantages:
        - it's continuous and differentiable
        - Combines MAE and MSE.
    - Requires tuning $\delta$

5. Sparse Entropy Loss: designed for image segmentation, where one-hot encoding is used in model output (after softmax) but training data only has labels. The loss per pixel is $l = -log(p_{true})$

For example:

For 3 pixels, training data labels are: `[2,1,0]`

Output data:

$$
\begin{gather*}
0.1 & 0.7 & 0.2 \\
0.3 & 0.7 & 0.0 \\
0.5 & 0.4 & 0.1 \\
\end{gather*}
$$

Then, the loss for each pixel is:

$$
\begin{gather*}
-log(0.2) & -log(0.3) & -log(0.5)
\end{gather*}
$$

You can add up the losses and get the sum of it

========================================================================

## Classification Losses

========================================================================

### Cross Entropy Loss (Log Loss)

$$
-\frac{1}{n} \sum (y_ilog(\hat{y}_i) + (1-y^i)log(1-\hat{y}_i))
$$

- Advantages:
  - Good for classification problems.
  - Suitable for probablistic Interpretation, and penalizes wrong confident predictions heavily
- Disadvantages:
  - when $y^i$ is (0,1). And the closer it is to the bounds, the loss would be larger

### Negative Log-Likelihood Loss (NLL)

Given a number of true classes (like acceptable output words in a sentence generator), NLL measures how likely we get any true class based on our prediction, a log-probability distribution across all outputs.

- Input Requirements: The model's output should be log-probabilities when using NLL Loss.
- Target Format: The targets should be provided as class indices, not one-hot vectors.

$$
\begin{gather*}
NLL = \sum_I -log(\text{true class}_i | {prediction})
\end{gather*}
$$

- For M samples,

$$
\begin{gather*}
\text{NLL} = \frac{1}{m} \sum_M(\text{NLL}_m).
\end{gather*}
$$

NLL is always used with log-softmax activations to ensure numerical stability.

#### NLL Example

1. We are given a single example, with logits `logits=[1.0, 2.0, 0.5, ]`
2. Apply log-softmax $log(\frac{exp(x_i)}{\sum_i exp(x_i)})$ and get `[−1.46303, −0.46303, −1.96303, ]`
3. Given a single true class `I=[1]`, NLL becomes

$$
\begin{gather*}
\text{NLL} = \sum_I -log(\text{true class}_i | \text{prediction}) = -log(y_1 | \text{prediction}) = 0.46303
\end{gather*}
$$

========================================================================

## Multicalss Classification & Image Segmentation Losses

========================================================================

Multiclass classification and image segmentation tasks both suffer the **data imbalance problem**. E.g., the background in image segmentation, or negative classes in multi-class classification make up around 80%. Do we want to count the accuracies in those areas? Yes. But do we want to treat them equally as the positive object classes? No. Because the positive object classes are fewer, which makes our model more sensitive them.

### IoU and Dice Loss

- IoU Loss
    IoU loss is: `1 - IoU`
    <div style="text-align: center;">
    <p align="center">
        <figure>
            <img src="https://github.com/user-attachments/assets/212cfcc3-b6cb-49bd-9197-9b1c850147e1" height="300" alt=""/>
            <figcaption><a href="https://www.google.com/url?sa=i&url=https%3A%2F%2Fkorlakuntasaikamal10.medium.com%2Fintersection-over-union-a8e04c3d03b3&psig=AOvVaw1Goh-uTS9ihTL4MTDw-fX2&ust=1723930055452000&source=images&cd=vfe&opi=89978449&ved=0CBcQjhxqFwoTCIijnYm6-ocDFQAAAAAdAAAAABAE">Source: Kamal_DS </a></figcaption>
        </figure>
    </p>
    </div>

- Dice Loss is $1 - \frac{2overlap}{\text{total\_area}}$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/35686620-3da7-45bb-abf7-b86a5498afd7" height="300" alt=""/>
        <figcaption><a href="https://cvinvolution.medium.com/dice-loss-in-medical-image-segmentation-d0e476eb486">Source</a></figcaption>
    </figure>
</p>
</div>

When training accuracy (overlap) increases, IoU loss is more sensitive to it at the very beginning, but not as sensitive when training accuracy is already high. Comparatively, Dice loss is more sensitive in high training accuracies, largely due to `2*overlap`. This can be proven by taking the derivative of the two losses.

My implementation:

```python
intersect = (pred == labels)
dice_loss = 1 - 2 * (intersect.sum().item())/ (pred.shape + labels.shape)
```

Along with weighted cross entropy loss, Dice loss (or the sensitivity) function were Introduced by Sudre in 2017 [1].

### Focal Loss

Focal loss addresses the class imbalance problem by penalizing the correctly idenfied classes. $\gamma$ is a hyperparameter, $p_t$ is the probability of correct classification.

$$
\begin{gather*}
FL = (1-p_t)^{\gamma} (-log(p_t))
\end{gather*}
$$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/c551930b-edc4-4cf5-b11c-892367381e70" height="300" alt=""/>
    </figure>
</p>
</div>

The above is for image segmentation, where every pixel has **only one true label**. For multi-class classification problems, one can use `FocalLoss` as well, but with some adaptations. [This post is very well written](https://www.kaggle.com/code/thedrcat/focal-multilabel-loss-in-pytorch-explained)

```python
import torch
torch.set_printoptions(precision=4, sci_mode=False, linewidth=150)

num_label = 8
logits = torch.tensor([[-5., -5, 0.1, 0.1, 5, 5, 100, 100]])
targets = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1]])
logits.shape, targets.shape

# will all one be better?
def focal_binary_cross_entropy(logits, targets, gamma=1):
    """
    The logic of the below is: 
        - For False Positive: adjusted_p = 1-p (~0), 1-adjusted_p ~1 -> logp = -log(adjusted_p) (far larger than 0) 
            -> loss = logp * (1-adjusted_p)**gamma = large number ✅
        - For False Negative: adjusted_p = p (~0), 1-adjusted_p ~1 -> logp = -log(adjusted_p) (far larger than 0) 
            -> loss = logp * (1-adjusted_p)**gamma = large number ✅
        - For True Negative: adjusted_p = 1-p (~1), 1-adjusted_p ~0 -> logp = -log(adjusted_p) (~0 ) 
            -> loss = logp * (1-adjusted_p)**gamma = ~0 ✅
        - For True Positive: adjusted_p = p (~1), 1-adjusted_p ~0 -> logp = -log(adjusted_p) (~0 ) 
            -> loss = logp * (1-adjusted_p)**gamma = ~0 ✅
    - torch.clamp() is in place to avoid overflow or under flow
    """
    l = logits.reshape(-1)
    t = targets.reshape(-1)
    p = torch.sigmoid(l)
    p = torch.where(t >= 0.5, p, 1-p)
    print(f"adjusted probability: {p}")
    print(f"1-adjusted_p: {1-p}")
    logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
    print(f"logp: {logp}")
    loss = logp*((1-p)**gamma)  #[class_num]
    print(f'If the starting index = 1, indices 2, 3, 5, 7 should have a large loss: {loss}')
    loss = num_label*loss.mean()
    return loss

focal_binary_cross_entropy(logits, targets)
```

## References

[1] Sudre, C. H., Li, W., Vercauteren, T., Ourselin, S., & Cardoso, M. J. Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations. Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support: 3rd International Workshop, DLMIA 2017, and 7th International Workshop, ML-CDS 2017 Held in Conjunction with MICCAI 2017, Quebec City, QC, Canada, September 14, 2017, Proceedings, pp. 240–248. Springer, 2017.
