---
layout: post
title: Deep Learning - Object Detection Notes Part 2
date: '2022-02-13 13:19'
subtitle: R-CNN
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Region Based CNN (R-CNN, Girshick et al. CVPR 2014)

[Zhihu](https://zhuanlan.zhihu.com/p/383167028)

Regional Proposal is the core of R-CNN. It first uses a segmentation algorithm to find regions with objects, then use these regions as "region proposals" for CNN to run on [2]. Each region outputs `[label, bounding box]`. 

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/e48cefc4-d7b9-4bb2-9a03-5a8ecebeff45" height="200" alt=""/>
    </figure>
</p>
</div>

1. Use Selective Search Algorithm to come up with 2000 region proposals: TODO
    - Use Hierarchical Grouping Algorithm  (Felzenszwalb and Huttenlocher, 1999)
        TODO: https://zhuanlan.zhihu.com/p/39927488
2. Use AlexNet for Feature Extraction on 2000 region proposals.
    - Get 2000x4096 feature vector
3. Classification & bounding box
    - Use 21 SVM to classify 21 classes (including background) on 2000 region proposals
        - Each SVM has 21 values.
    - Parallel to classication, use TODO regression for bounding box regression

Later came Fast R-CNN (Girshick, ICCV 2015). Fast R-CNN propose regions, then uses convolution implementation of sliding windows to classify all proposed regions.

Then came Faster R-CNN (Ren, He et al. NeurlPS 2015). They are all slower than YOLO. From Prof. Andrew Ng's perspective, YOLO's 1-stage architecture is more concise.

- YOLOv3: 30 fps on high end CPU
- Faster R-CNN: 7 fps+
- YOLOv4 and YOLOv5: 60fps+

## References
- [1] [Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016 (pp. 779-788). IEEE.](https://arxiv.org/pdf/1506.02640)
- [2] [R. Girshick, J. Donahue, T. Darrell, and J. Malik, "Region-Based Convolutional Networks for Accurate Object Detection and Segmentation," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014, pp. 580–587.](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)
