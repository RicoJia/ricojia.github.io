---
layout: post
title: RGBD SLAM Bundle Adjustment Part 2
date: '2024-07-12 13:19'
excerpt: RGBD SLAM Backend Introduction
comments: true
---

If you haven't, please check out the previous article on [how to formulate SLAM as an optimization problem](./2024-07-11-rgbd-slam-bundle-adjustment.markdown)

## How To Formulate SLAM Optimization Into Pose Graph

What are nodes: each node represents a pose of the robot's trajectory. Each edge between two edges represent the relative position of the two poses.

A graph G is composed of vertices (V) and edges (E) $G={V,E}$. An edge can connect to 1 vertex (unary edge), 2 vertices (binary edge), or even multiple vertices (hyper edge). Most commonly, a graph has binary edges. But when there are hyper edges, this graph is called "hyper graph".

TODO: on what graph optimization does using LM, and in Block?

## Robust Kernels

In SLAM, it's common to have mismatched feature points. In that case, we add an edge that we shouldn't have added between a camera pose and a 3D point. The wrong edge could give huge error and high gradient, so high that just optimizing parameters associated could yield more gradient than the correct edges. One way to make up for it is to regulate an edge's cost, so it doesn't get too high nor gives too high of a gradient. Huber loss, cauchy loss are common examples. In SLAM terminology, loss is also called  "kernel". (How come people in different computer science disciplines love corn so much?)

For example, Huber kernel switches to first order when error is higher than $\delta$. Also, it is continuous and derivable at $y - \hat{y} = \delta$. This is important, because we need to get gradient everywhere at the cost function.

$$
\begin{gather*}
e=
\begin{cases} 
\frac{1}{2} x^2 & \text{for } |x| \le \delta \\
\delta (|x| - \frac{1}{2} \delta) & \text{for } |x| > \delta 
\end{cases}
\end{gather*}
$$

### My Implementation