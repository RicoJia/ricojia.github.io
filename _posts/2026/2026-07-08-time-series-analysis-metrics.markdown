---
layout: post
title: "[Robotics] Time Series Analysis Metrics"
date: 2026-07-08 13:19
subtitle: MAD
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Career Advice
---

## 1 - MAD

There are **two different definitions** of MAD. In machine learning papers, you have to check which one they mean.

### 1- 1 Mean Absolute Deviation

Given measurements

$$
x = [3,4,5,8]
$$

Mean:

$$
\mu = 5
$$

Absolute deviations:

$$
|x-\mu| = [2,1,0,3]
$$

MAD is

$$
\frac{2+1+0+3}{4}=1.5
$$

### 1 - 2 Median Absolute Deviation (more common for robust statistics)

Median = 4.5

Absolute deviations:

$$
[1.5,0.5,0.5,3.5]
$$

Median of those:

$$
MAD=1.0
$$

---

# 2. Lag :  delay between an observed _data_ point and its preceding values

## 2 - 1 Definitions

Lag simply means **looking one time step into the past**. Suppose your boat probability is

| Frame | Probability |
| ----- | ----------: |
| 1     |        0.10 |
| 2     |        0.20 |
| 3     |        0.75 |
| 4     |        0.85 |
| 5     |        0.82 |

Lag-1 is simply:

$$
x_{t-1}
$$

```text
Current frame     Previous frame

0.20      ← 0.10

0.75      ← 0.20

0.85      ← 0.75

0.82      ← 0.85
```

Lag-2:

$$
x_{t-2}
$$

Example

| Current | Lag-1 | Lag-2 |
| ------: | ----: | ----: |
|    0.75 |  0.20 |  0.10 |
|    0.85 |  0.75 |  0.20 |
|    0.82 |  0.85 |  0.75 |

### 2 - 2 Why are lag features useful?

Imagine the point count from the boat.

```text
Frame

500

503

498

505

510
```

Very stable.

Now a random clutter patch

```text
120

40

250

70

180
```

Highly unstable.

The relationship between is very different.

```text
current

vs

lag-1

vs

lag-2
```

A classifier can learn that.

### 2 - 3 Another common meaning: lag correlation

Sometimes papers don't mean the lagged value itself.

They mean the **autocorrelation**.

For lag-1,

$$
Corr(x_t,x_{t-1})
$$

For lag-2,

$$
Corr(x_t,x_{t-2})
$$

Suppose your point count is

```text
500
501
502
503
504
```

Lag-1 correlation is almost `1` because every frame looks like the previous one.  Random noise

```text
20
400
90
310
15
```

has a lag-1 correlation near `0`

---

## 3 - Coefficient of Variation (CV)

**CV here means coefficient of variation, not covariance.**

$$  
CV=\frac{\text{standard deviation}}{\text{mean}}  
$$

It is dimensionless. For example, if sonar intensity has mean (100) and standard deviation (45),

$$  
CV=\frac{45}{100}=0.45  
$$

Intensity fluctuates by roughly 45% of its mean.
