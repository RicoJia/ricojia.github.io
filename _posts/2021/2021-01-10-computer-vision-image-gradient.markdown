---
layout: post
title: Computer Vision - Image Gradient
date: '2021-01-10 13:19'
subtitle: Sobel Operator, Signed Distance Field
comments: true
header-img: "home/bg-o.jpg"
tags:
    - Computer Vision
---

Image gradients capture the rate of intensity change in an image and are fundamental for detecting edges.

## Sobel Operator

The Sobel operator is a popular edge detection filter that computes **image gradients by combining smoothing with differentiation**. It **approximates the gradient** of an image's intensity using **two 3×3 kernels**—one for the horizontal (X) direction and one for the vertical (Y) direction—derived from a smoothing filter (typically [1, 2, 1]) and a differentiation filter (typically [-1, 0, 1]).

- X Direction: the Sobel kernel for the X derivative is formed by applying a smoothing filter along the vertical axis and a differentiation filter along the horizontal axis. With a normalization factor of 1/8, it is given by:

```
[ -1,  0,  1 ]
[ -2,  0,  2 ] * 1/8
[ -1,  0,  1 ]
```

- Y Direction: for the Y derivative, the kernel uses the same concept, but with the roles reversed (note that the sign convention for the Y axis might be inverted relative to X):

[  1,  2,  1 ]
[  0,  0,  0 ] * 1/8
[ -1, -2, -1 ]


### Application:

- Convolution: the original image A is convolved with these kernels (using correlation or, equivalently, convolution with a flipped kernel) to compute the horizontal and vertical gradients:

```
Ax=Gx∗A
Ay=Gy∗A
```

Gradient Magnitude and Direction:
The overall gradient magnitude GG is computed as:

$$
\begin{gather*}
\begin{aligned}
& G = \sqrt{A_x^2 + A_y^2}
\end{aligned}
\end{gather*}
$$

The gradient direction is determined by:

$$
\begin{gather*}
\begin{aligned}
& \theta = atan2(A_y, A_x)
\end{aligned}
\end{gather*}
$$

### Noise Reduction

In the real world, noise often produces many **high gradients** that do not correspond to actual edges. To combat this, image processing pipelines typically **apply a smoothing filter before computing gradients**. This smoothing is distinct from the implicit smoothing in operators like Sobel.

The key idea is based on the associative property of differentiation:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://user-images.githubusercontent.com/39393023/131610963-95e94f24-2c92-4ecf-8ade-941fcfa51e56.png" height="200" alt=""/>
    </figure>
</p>
</div>

Those noise would yield:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://user-images.githubusercontent.com/39393023/131610964-cb088b8e-d070-4fa5-83bd-b617d0cd1a7d.png" height="300" alt=""/>
    </figure>
</p>
</div>


Therefore we need to smooth the pic first: (take x direction for example) (so this is different smoothing than sobel operator)
      ```
      del(h * f)/del(x) = del(h)/del(x) * f
      ```

- smoothing function h, and h' are, which will save us from a whole differentiation cycle. 
- This is because differentiation allows associativity, and associaitivity allows us to get direvative of the filter.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://user-images.githubusercontent.com/39393023/131610967-05e9b3f8-5875-45b6-9b97-1b2b8b7c8f79.png" height="300" alt=""/>
    </figure>
</p>
</div>

How do we know where there's a peak? Use the second derivative. We can identify extremas 

<p align="center">
<img src="https://user-images.githubusercontent.com/39393023/131610971-65d34584-0419-4d8d-83cc-df9f19703b9c.png" height="200" width="width"/>
<figcaption align="center">Second Order Derivative</figcaption>
</p>

To visualize the first order derivative of the fitlers: 

<p align="center">
<img src="https://user-images.githubusercontent.com/39393023/131610972-de2d16d2-eeda-4726-ba71-132c5bd9cc7c.png" height="300" width="width"/>
<figcaption align="center">Visualization of First Order Derivative of Smoothing Filters</figcaption>
</p>

The bigger the smoothing filter, the less noise will be incorporated

<p align="center">
<img src="https://user-images.githubusercontent.com/39393023/131610973-d202737c-262b-481d-813a-2aedf8f20589.png" height="200" width="width"/>
<figcaption align="center">Smoothing Filter</figcaption>
</p>

## Signed Distance Field

Signed Distance Field of a cat:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/eaf6881d-a9a5-4bb5-be10-00122c0b3c3c" height="300" alt=""/>
        <figcaption><a href="https://shaderfun.com/2018/07/23/signed-distance-fields-part-8-gradients-bevels-and-noise/">Source</a></figcaption>
    </figure>
</p>
</div>

With signed-distance-field, we can create a "fake" 3D effect on the cat:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/542c8e6b-7b09-47fe-b8bd-531bcdf775cc" height="300" alt=""/>
        <figcaption><a href="https://shaderfun.com/2018/07/23/signed-distance-fields-part-8-gradients-bevels-and-noise/">Source</a></figcaption>
    </figure>
</p>
</div>

A Signed Distance Field (SDF) represents surfaces implicitly by storing the shortest distance to the nearest surface at each point in space. While SDFs enable smooth surface reconstruction, they are not commonly used in industry due to computational complexity.

Instead, Surfel-based methods are more practical, representing surfaces as discrete surface elements (surfels)—small patches that approximate local geometry.

Here is a video of a surfel based method

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/fea4498d-9ba6-4ec6-8c78-ecc7c06b8f60" height="300" alt=""/>
            <figcaption><a href="https://www.youtube.com/watch?v=2gZNpFE_yI4">Video:  Real-time Scalable Dense Surfel Mapping (2018)</a></figcaption>
       </figure>
    </p>
</div>
