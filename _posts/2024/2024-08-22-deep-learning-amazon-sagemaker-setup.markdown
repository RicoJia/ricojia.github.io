---
layout: post
title: RGBD SLAM - Amazon SageMaker Usage Notes
date: '2024-08-22 13:19'
subtitle: Amazon SageMaker, EC2 Instances
header-img: "img/post-bg-unix"
tags:
    - Deep Learning
comments: true
---

## Components

- Debugger (At least compatible with TensorFlow). [Video Explanations](https://youtu.be/MqPdTj0Znwg)

  - can add rules, e.g., vanishing gradient rule (your model is not updating anymore), overfit (can see in trial details about warnings)
  - Save the model on S3 automatically
  - github amazon sagemaker  examples
  - Pre-req: load Amazon SageMaker Studio

- Pre-Installed Frameworks
  - XGBoost
  - [Autogluon](https://docs.amazonaws.cn/sagemaker/latest/dg/semantic-segmentation.html):
    - Deployment
      - these three all have a backbone (encoder), and a decoder
    - Models:
      - FCN
      - PSP
      - Deep Lab V3
    - Process:
            1. Put Data on S3
            2. Org data into: train (jpg), validation (jpg), train_annotation(png), validation_annotation (single channel uint8 png), label_map (json, mapping between uint8 -> label names)
  - Pytorch

I [like Julien Simmon's videos on Amazon SageMaker.](https://www.youtube.com/watch?v=sOUhLiI85sU&list=PLJgojBtbsuc0E1JcQheqgHUUThahGXLJT&index=3)

## EC2 GPU & CPU Instances

- Free tier 50 hours per month of m4.xlarge or m5.xlarge for training, 125 hours per month of m4.xlarge or m5.xlarge for hosting
- m5.xlarge, g5.xlarge pricing  [(Source: Reddit Post)](https://www.reddit.com/r/LocalLLaMA/comments/1dclmwt/benchmarking_inexpensive_aws_instances/), there is a performance and price chart as of Amazon EC2 types:

| Instance Type      | Cores | RAM     | GPU         | Prompt Eval Rate (tokens/s) | Eval Rate (tokens/s) | Price (per hr) | Price (per mo) |
|--------------------|-------|---------|-------------|-----------------------------|----------------------|----------------|----------------|
| c7g.8xlarge (CPU)       | 32    | 64 GB   | N/A         | 38.38                       | 25.07                | $1.27          | $941.16        |
| r6g.4xlarge (CPU)      | 16    | 128 GB  | N/A         | 10.15                       | 8.29                 | $0.88          | $657.10        |
| g4dn.xlarge (GPU)      | 4     | 16 GB   | 16 GB       | 222.23                      | 41.71                | $0.58          | $434.50        |
| g4dn.2xlarge (GPU)      | 8     | 32 GB   | 32 GB       | 214.25                      | 41.74                | $0.84          | $621.24        |
| g5.xlarge (GPU)         | 4     | 16 GB   | 24 GB       | 624.29                      | 68.08                | $1.12          | $831.05        |
| g5.2xlarge (GPU)        | 8     | 32 GB   | 24 GB       | 624.48                      | 66.67                | $1.35          | $1,000.96      |

- [more on amazon EC2 types (pricing info has to be requested)](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-instance-types.html)
