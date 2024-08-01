---
layout: post
title: RGBD SLAM Bundle Adjustment Part 2
date: '2024-07-12 13:19'
excerpt: RGBD SLAM Backend Introduction
comments: true
---

If you haven't, please check out the previous article on [how to formulate SLAM as an optimization problem](./2024-07-11-rgbd-slam-bundle-adjustment.markdown)

## How To Formulate SLAM Optimization Into Graph

What are nodes: each node represents a pose of the robot's trajectory. Each edge between two edges represent the relative position of the two poses.

### What is Graph Optimization

A graph G is composed of vertices (V) and edges (E) $G={V,E}$. An edge can connect to 1 vertex (unary edge), 2 vertices (binary edge), or even multiple vertices (hyper edge). Most commonly, a graph has binary edges. But when there are hyper edges, this graph is called "hyper graph"

TODO: on what graph optimization does using LM, and in Block?