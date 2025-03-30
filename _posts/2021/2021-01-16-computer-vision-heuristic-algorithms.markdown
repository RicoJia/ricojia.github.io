---
layout: post
title: Computer Vision - Heuristic Algorithms
date: '2021-01-13 13:19'
subtitle: K Means
comments: true
header-img: "home/bg-o.jpg"
tags:
    - Computer Vision
---

## Introduction

In mathematical optimization and computer science, heuristic is a technique designed for problem solving more quickly when classic methods are too slow for finding an exact or approximate solution, or when classic methods fail to find any exact solution in a search space. 

## K Means

Example:Given a list of points: `(2,3),(5,4),(3,8),(8,8),(7,2),(6,3)(2,3),(5,4),(3,8),(8,8),(7,2),(6,3)`, with K=2 clusters

1. Randomly pick two centroids: `(2,3)`, `(8, 8)`
2. For each point, compare its distance to each centroid. Get 
    ```
    (2, 3) → Cluster 1
    (5, 4) → Cluster 1
    (3, 8) → Cluster 2
    (8, 8) → Cluster 2
    (7, 2) → Cluster 1
    (6, 3) → Cluster 1
    ```
3. Find new centroids of each centroid, by calculating the new averages:
    ```
    (2+5+7+6)/4,(3+4+2+3)/4=(5,3)
    (3+8)/2,(8+8)/2=(5.5,8)(3+8)/2,(8+8)/2=(5.5,8)
    ```

4. Repeat Assignment. If the result is the same, then the algorithm converges.

There's a chance kmeans could alternate. In that case, need to initialize differently.