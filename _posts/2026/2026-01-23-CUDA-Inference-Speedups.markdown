---
layout: post
title: "[CUDA - 6] CUDA Inference Speedups"
date: 2026-01-21 13:19
subtitle: inference mode
comments: true
header-img: img/post-bg-alibaba.jpg
tags:
  - CUDA
---
`torch.backends.cudnn.benchmark = True` tells PyTorch/cuDNN to try multiple GPU convolution algorithms and cache the fastest one for the input shapes it sees. This can speed up CNN-style models when input sizes are mostly fixed, but can hurt performance if shapes change constantly because cuDNN may keep re-benchmarking.

`torch.inference_mode()` disables autograd bookkeeping during inference. Compared with normal execution, PyTorch does not build gradient graphs, and it also disables extra tracking such as view tracking and version counter updates. That usually means **lower memory use** and **faster execution** for pure prediction code. PyTorch notes that `inference_mode` does **not** automatically call `model.eval()`, so you still need `model.eval()` for dropout/batchnorm behavior.
