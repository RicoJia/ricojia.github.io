---
layout: post
title: "Model Parameter vs VRAM"
date: 2026-03-05 13:19
subtitle: VRAM
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine-Learning
---

VRAM = Video RAM. It is the memory on your GPU, separate from your normal computer RAM. The GPU uses VRAM to store:

- model weights
- input images / batches
- intermediate activations
- gradients during training
- optimizer states during training
- temporary CUDA/TensorRT buffers

For inference/training, GPU memory has several components:

```text
Total VRAM ≈ weights + activations + gradients + optimizer states + temporary buffers + input/output tensors
```

### 1. Model parameter memory is linear

Weights scale almost perfectly linearly with parameter count:

| Precision   | Memory per parameter | Example: 100M params |
| ----------- | -------------------: | -------------------: |
| FP32        |              4 bytes |              ~400 MB |
| FP16 / BF16 |              2 bytes |              ~200 MB |
| INT8        |               1 byte |              ~100 MB |
| INT4        |             0.5 byte |               ~50 MB |

So for **inference weights only**:

```text
weight memory = parameter count × bytes per parameter
```

For inference, especially CNNs/ViTs/detectors, **activation memory** can dominate. It depends on:

```text
batch size × image resolution × feature map sizes × number of layers
```

**So two models with the same parameter count can use very different VRAM if one has larger intermediate feature maps.**

Example: a detector with a large high-resolution neck/FPN can use more VRAM than another model with more parameters but smaller intermediate tensors.

### 2. Training VRAM is much less linear

Training adds:

* weights
* gradients
* optimizer states
* saved activations for backprop
* augmentation / dataloader staging
* loss buffers

For Adam/AdamW in mixed precision, a rough rule is:

```text
training parameter memory ≈ 12–18 bytes per parameter
```

before activations.

So a 100M-param model might need **1.2–1.8 GB just for parameter-related training state**, then activations can add much more.

For **YOLO-style detection training**, VRAM is usually more sensitive to:

```text
image size > batch size > model size
```

**Parameter count matters, but changing from `640` to `1280` image size can blow up activation memory much faster than moving from a small to medium model.**

For **inference**, model size is more predictive, but resolution and backend still matter.  So the relationship is:

```text
Weights vs params: linear
Total VRAM vs params: only loosely correlated
Training VRAM vs params: partly linear, often dominated by activations
```
