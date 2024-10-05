---
layout: post
title: Deep Learning - Face Recognition Prelude
date: '2022-02-23 13:19'
subtitle: Face Frontalization
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Introduction

DeepFace introduced a 3D alignment step that projects 2D face images into a frontal view [1]. This is called "frontalization". A very cool 2D->3D problem. A frontal view is a view right in front of the face. When given a side view of the face, the face is warped. Some main methods include:

    - 2D frontalization 
        - Detects 6 fiducial points: center of the eye, tip of the nose, etc. to find [scale, rotation, and translation] of something to generate a 3D aligned crop?? 
        - out-of-plane rotation? What is that?
    - 3D alignment is handcrafted?


Did you create a 3D model of faces?


[1] [Taigman, Y., Yang, M., Ranzato, M. A., & Wolf, L. 2014. DeepFace: Closing the Gap to Human-Level Performance in Face Verification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014, 1701-1708. DOI: https://doi.org/10.1109/CVPR.2014.220](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)

