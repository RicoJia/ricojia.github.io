---
layout: post
title: Computer Vision - Warping
date: '2021-01-07 13:19'
subtitle: Affine Transformation, Perspective Warping, Non-Linear Warping.
comments: true
header-img: "home/bg-o.jpg"
tags:
    - Computer Vision
---

## Introduction

Warping is a process to transform pixels to another shape using a mapping function. There are three kinds of warping: affine warping, perspective warping, and non-linear warping. They are used in image-stitching, optical flow, and 2D->3D texture mapping.
 
## Affine Warpping

Affine warping include: shearing, scaling, translation, and rotation

Translation is simply mapping objects from one 2D pixel location to another. 
The effect of that is to keep the image window size, but shift the image to the right by 200 pixels, and 30 pixels down.

```python
import numpy as np
import cv2
tx = 200; ty=30
M = np.float32([
[1, 0, tx],
[0, 1, ty]
])
dst = cv2.warpAffine(img,M,(cols,rows))
```

- `cv2.warpAffine(img,M,(cols,rows))` third arg is the size of the output image

See the black space?

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/5e8bfbe0-d48f-4ec3-8934-629dc7a80345" height="200" alt=""/>
    </figure>
</p>
</div>

However, the `cv2.warpAffine` is for general affine operation: 

$$
\begin{gather*}
\begin{bmatrix}
\alpha & \beta & (1 - \alpha) \cdot \text{center.x} - \beta \cdot \text{center.y} \\
-\beta & \alpha & \beta \cdot \text{center.x} + (1 - \alpha) \cdot \text{center.y}
\end{bmatrix}
\end{gather*}
$$

Where

$$
\begin{gather*}
\alpha = scale * cos(\theta)
\\
\beta = scale * sin(\theta)
\end{gather*}
$$

So we **need to plug in the image center coords into this matrix.** if we want to rotate the image by any arbitrary point


THe way I interpret this is:

- OpenCV has an underlying function for rotating the image by the top left corner. Say points in the image frame $\vec{P}=(px, py)$. After rotation, points becomes $R \vec(P)$
- However, we want to rotate points by the chosen center frame $\vec{C}$, then find their image frame coordinates. We denote the chosen center frame coordinates as $\vec{X} = \vec{P} + \vec{C}$
- So we want to find linear translation $\vec{T}$ so after rotating points in the image frame, we shift the points so they are rotated as if by the chosen center. So that becomes: 

$$
\begin{gather*}
R\vec{P} + \vec{T} = R(\vec{P} - \vec{C}) + \vec{C}
\\ => \vec{T} = (I - R) \vec{C}
\end{gather*}
$$

$(I - R)$ corresponds to the formula in the function

## Perspective Warping include

Perspective warping changes perspective as if viewing the object from a different angle.

TODO

## Non-Linear Warping

This can be used in lens distortion correction.