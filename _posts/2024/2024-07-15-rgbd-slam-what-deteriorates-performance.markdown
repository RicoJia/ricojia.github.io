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

