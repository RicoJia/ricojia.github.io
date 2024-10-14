---
layout: post
title: Computer Vision - ORB Features And Matching
date: '2021-01-22 13:19'
subtitle: This Blog Shows The Theory and Implementation From Scratch of ORB Features
comments: true
tags:
    - Computer Vision
---

## Intro

In SLAM and Computer Vision, distinguishing the same points across multiple camera views is necessary for visual odometry. These "distinguishable points" are called features. In general, we want our feature detection to be robust to:

- General camera motions, like Rotation, i.e., after rotating the camera, the same point can still be detected
- Illumination changes

Corners and edges are distinguishable. Before 2000, there are Harris Corner, FAST corner, etc. However, those are not sufficient especially in the case when the camera rotates. Therefore, methods like SIFT (David Lowe, ICCV 1999), SURF (Bay et al. ECCV 2004), ORB (Rublee et al. ICCV 2011) came about [1]. These methods are composed of a keypoint, and a feature descriptor. SIFT (Scale-Invariant-Feature Transform) is most classic and considers changes in illumination, scale, rotation during image transformation. However, it comes with a significant computation cost which makes it not fast enough for a real-time system (but fast enough with GPU acceleration).

FAST keypoint is very fast to calculate, but it does not have a descriptor. ORB (Oriented FAST and Rotated BRIEF) uses a rotated BRIEF descriptor on top of the FAST detector. In some experiments, it can be over 300x faster than SIFT, while its performance is still decent. So ORB is a fair trade-off in SLAM.

## ORB Features (Oriented FAST and Rotated BRIEF)

<p align="center">
<img src="https://github.com/RicoJia/The-Dream-Robot/assets/39393023/90f44985-6c87-4a1d-8663-462b74e4b651" height="300" width="width"/>
</p>

## ORB Feature Detection Workflow

1. Generate an image pyramid of n level
    1. Gaussian Blur the image because ORB is sensitive to noise

2. Fast Feature Identification at a specific image pyramid level
    1. Find a pixel. Draw a circle with a radius of 3 pixels (16 pixels)
    2. If you can find **more than N consecutive pixels** with an intensity greater than T, then this pixel is a keypoint.
        - N=12 is FAST-12. For that, you can check pixel 1,5,9,13 to rule out.
    3. Repeat this on other pixels
    4. Apply non-maximal suppresion to avoid keypoint clustering

3. Orientation Identification (Not in FAST)
    - Select a patch that centers the corner.
    - Compute the image moments of the patch, then compute the intensity centroid. Use the center of the patch as the origin $O$
        $$
        m_{pq} = \sum_{x,y} x^{p} y^{q} I(x,y)
        \\
        C = [\frac{m_{10}}{m_{00}}, \frac{m_{01}}{m_{00}}]
        $$
        - So, $x, y \in [-3, 3]$ for the entire image patch. $p, q \in [0,1]$ for the zeroth and first order image moments.
    - Calculate orientation of $\vec{OC}$ using **the quadrant-aware** atan2
        $$
        \theta = atan2 (m_{01}, m_{10})
        $$

4. Represent the descriptor in steered BRIEF
    - Pre-select 256 pairs of pixels in the patch.
        - In Rublee et al's paper, a large number of images were used to compare the image intensitiy of these selected pixels of keypoints. 
        They mapped out the mean of all these image pairs, and the correlation within each pair.
    - Steer the descriptor. From the "normalized image" to the rotated one (the patch we have):
        $$
        S = R(\theta)[x_1,y_1; ... x_n, y_n]
        $$
    - Compare their intensities. If $I(p1) > I(p2)$ then you get a 1, else 0. Put it in a 256-bit array

5. Repeat the same process in an image pyramid for scale invariance. This is needed because some features are not ORB features anymore if you zoom in/out.

### Fast Library For Approximate Nearest Neighbor (FLANN)

There are two potentially parallel ways to find an approximate nearest neighbor, when **number of dimensions is high**. The traditional KD Tree search will degrade largerly in that situation. This is an alternative is called LSH (Locality Sensitive Hashing), also a probablility datastructure for finding an approximate nearest neighbor
    - Need to specify a precision.

1. Randomized KD Tree.
    - Find D number of dimensions with the highest variances.
    - Each randomized KD tree has a randomly selected number of dimensions among these D dimensions, or split points. 
    - One can instantiate multiple randomized KD tree so each tree has a suboptimal tree. You can search them in parallel, too. 
    - Termination condition: if a pre-determined number of nodes are visited. (Precision?)

2. K means trees:
    - There are multiple levels. Each level has K clusters. 
    - Search: have a priority queue. Everytime you go to a level, add other branches to this priority queue.
    - If a pre-determined number of nodes are visited

3. Use Lowe's ratio to filter out good matches, if `(first_neighbor_hamming_dist)/(second_neighbor_hamming_dist) < 0.7`

## Implementation

Here is the [OpenCV implementation](https://github.com/barak/opencv/blob/051e6bb8f6641e2be38ae3051d9079c0c6d5fdd4/modules/features2d/src/orb.cpp#L533). The algorithm applies below tricks:

Here is [my own implementation](https://github.com/RicoJia/dream_cartographer/tree/main/rgbd_slam_rico/include/rgbd_slam_rico_exercises)

In step 3 orientation computation, OpenCV uses
    - Integral Image (a.k.a summed area table). There's a [leetcode question](https://leetcode.com/problems/range-sum-query-2d-immutable/description/) for it. Give it a try!

## References

[1] Rublee, E., Rabaud, V., Konolige, K., and Bradski, G. 2011. ORB: an efficient alternative to SIFT or SURF. In Proceedings of the 2011 International Conference on Computer Vision (ICCV '11). IEEE Computer Society, Washington, DC, USA, 2564-2571. DOI:https://doi.org/10.1109/ICCV.2011.6126544
