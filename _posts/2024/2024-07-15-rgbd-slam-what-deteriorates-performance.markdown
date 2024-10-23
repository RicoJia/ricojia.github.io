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

[In Roh et al.'s MBA-VO paper (ICCV 2021)](https://doi.org/10.1109/ICCV48922.2021.01178), two deep learning-based deblurring methods were evaluated:

    - A slower RNN with better quality [Tao et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tao_Scale-Recurrent_Network_for_CVPR_2018_paper.pdf).
    - A faster, but slightly less accurate, CNN (Kupyn et al., 2019).

For my RGBD SLAM, I chose the multiscale RNN due to its effectiveness in offline applications. This method creates an image pyramid, applies deblurring at multiple resolutions, and avoids overfitting through weight sharing.
    -  Tao et al. found that CNN with independent params could overfit.
