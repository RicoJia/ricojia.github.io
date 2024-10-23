---
layout: post
title: Deep Learning - Auto Differentiator From Scratch
date: '2022-01-06 13:19'
subtitle: Auto Diff Is The Dark Magic Of All Dark Magics Of Deep Learning
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Introduction

- Gradients here refer to scalar to matrix gradient.
- We need to accumulate gradients for mini-batch training.

Elementwise Multiplication gradients:
A * B = C

- del C / del A_ij = B_ij -> del C/ del A = B
