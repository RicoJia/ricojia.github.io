---
layout: post
title: Robotics - KNN Algorithms
date: '2024-1-9 13:19'
subtitle: ROS2, Waveshare
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
comments: true
---

## One-Pass Grid Method

## KD Tree

- Approximated nearest neighbor is d_split < alpha * d_max
- KD Tree without approximated nearest neighbor should be able to find all K nearest neighbors

### Optimized Hashing Function for Collision Checks

- `hash(x,y,z) = ( x p1 xor y p2 xor z p3) mod n`, where p1, p2, p3 are large prime numbers, in our case 73856093, 19349663, 83492791. [Reference](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf)
