---
layout: post
title: Deep Learning - Common Oopsies
date: '2022-05-17 13:19'
subtitle: Underflow
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Underflow

- `torch.softmax(X)` X is zero due to underflow.
