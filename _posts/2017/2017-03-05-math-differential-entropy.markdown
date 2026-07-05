---
layout: post
title: Math - Differential Entropy
date: 2017-03-05 13:19
subtitle:
comments: true
tags:
  - Math
---
## Differential Entropy

Differential entropy is the entropy version for a continuous random variable. For a discrete variable:

```
H(x) = -sum(p(x) * log(p(x)))

// continuous version
h(X) = -∫ p(x) * log(p(x)) dx
```

For a Gaussian

```
x ~ N(mu, Sigma)
```

Differential entropy

```
h(X) = ```
h(X) = h(X) = -∫∫∫ p(x, y, z) log(p(x, y, z)) dx dy dz
// along one dimension, 
p(x) = 1 / sqrt((2*pi)^3 * det(Sigma))
       * exp(-0.5 * (x - mu)^T * Sigma^{-1} * (x - mu))
         
``` 0.5 * log((2*pi*e)^d * det(Sigma))
```

over all 3D space.

where e is Euler's number `e = 2.718....` In C++: `e = std::exp(1.0)`

For 3D point distribution, Sigma is 3x3. e? Also note that x in meters vs x in millimeters will scale differential entropy. So differential entropy is not an absolute information amount

It's called differentail because "differential voluime element" along (dx, dy, dz). Remember, we are dealing with a continuoum, so it's probabilityb density, not actual probability
