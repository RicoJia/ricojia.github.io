---
layout: post
title: Math - Welford Algo for Cov Calculation
subtitle: 
date: '2017-02-05 13:19'
header-img: "img/bg-material.jpg"
tags:
    - Math
---

```cpp
template <typename Container, typename VectorType, typename Getter>
inline void compute_cov_and_mean(const Container &data, VectorType &mean, VectorType &cov, Getter &&getter) {
    const size_t len = data.size();
    assert(len > 1);
    
    // Initialize counters for the running mean and the sum of squared differences.
    size_t n = 0;
    mean = VectorType::Zero();
    VectorType M2 = VectorType::Zero();
    
    for (const auto &item : data) {
        const VectorType value = getter(item);  // get the point (Vec3d)
        ++n;
        // Compute the difference between the new value and the current mean.
        const VectorType delta = value - mean;
        // Update the mean.
        mean += delta / static_cast<double>(n);
        // Update the sum of squares of differences using the new value.
        const VectorType delta2 = value - mean;
        M2 += delta.cwiseProduct(delta2);
    }
    
    // For sample covariance (diagonal only), divide by (n - 1)
    cov = M2 / static_cast<double>(n - 1);
}
```
