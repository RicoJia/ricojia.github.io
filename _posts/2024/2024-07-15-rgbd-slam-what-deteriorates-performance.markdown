---
layout: post
title: RGBD SLAM - What Deteriorates Its Performance
date: '2024-07-15 13:19'
subtitle: Lessons Learned From My RGBD SLAM Project. Updates Come In Anytime
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - RGBD Slam
    - Deep Learning
comments: true
---

> “Yeah It's on. But wait, it doesn't look good.”

## Motion Blur

Motion blur could cause lower number of features and feature mismatching. These will further cause huge errors in PnP solving. Below is a scene of my room. Both images are blurred. See how many feature matches are missed and mismatched?


<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/740de502-c7c2-42c6-ab89-e35b3ddb4a19" height="400" alt=""/>
    </figure>
</p>
</div>

### Debluring

There are two types of methods: classical optimization methods, deep learning based methods. In [1], a deep deblurring network is used:

- slow but better quality: Xin Tao, Hongyun Gao, Xiaoyong Shen, Jue Wang, and Ji-aya Jia. Scale-recurrent network for deep image deblurring. In Computer Vision and Pattern Recognition (CVPR), 2018 - Choose this one
- Fast but a bit worse: Kupyn et al: rest Kupyn, Tetiana Martyniuk, Junru Wu, and Zhangyang Wang. Deblurgan-v2: Deblurring (orders-of-magnitude) faster and better. In International Conference on Computer Vision (ICCV), 2019

## References

[1]Junha Roh, Yeonkun Lee, and Ayoung Kim. 2021. MBA-VO: Motion Blur Aware Visual Odometry. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*. 11983-11992. https://doi.org/10.1109/ICCV48922.2021.01178
