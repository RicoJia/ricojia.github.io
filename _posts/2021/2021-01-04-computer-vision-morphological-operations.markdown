---
layout: post
title: Computer Vision - Morphological Operations
date: '2021-01-04 13:19'
subtitle: Erosion
comments: true
header-img: "home/bg-o.jpg"
tags:
    - Computer Vision
---

## Erosion

Slide a 3x3 kernel over every pixel of a binary image. If **all 9 pixels under the kernel are 1**, the center pixel stays 1. Otherwise, it becomes 0.

In the example below, the 3x3 blob shrinks to its single center pixel, because that's the only pixel whose whole 3x3 neighborhood is 1. The isolated pixel in the bottom-right corner disappears entirely. That's why erosion is useful for removing small noise and separating objects that barely touch.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/DzmZBQjH/download.jpg" height="200" alt=""/>
        <img src="https://i.postimg.cc/Df1zjw6y/download-(1).jpg" height="200" alt=""/>
        <figcaption>Up: before erosion, Down: after erosion</figcaption>
    </figure>
</p>
</div>
