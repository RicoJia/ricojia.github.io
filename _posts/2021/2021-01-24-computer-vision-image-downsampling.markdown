---
layout: post
title: Computer Vision - Image Downsampling
date: '2021-01-24 13:19'
subtitle: Bicubic, Bilinear, Nearest Neighbor Interpolation, Ringing Effect
comments: true
tags:
    - Computer Vision
---

## Introduction

**What is image downsampling?**: Image Downsampling is to reduce spatial resolution and make an image smaller.

**Why performing interpolation during downsampling?**: image downsampling essentially is to map larger areas of pixels onto smaller areas. Therefore, one pixel in the smaller image is determined by multiple pixels on the original image. Interpolation helps us find one value out of multiple pixels.

Image binning: Pixel binning is to combine adjacenet pixels into a super pixel. This can be done on the hardware level on CMOS (Complementary-Metal-Oxide-Semiconductor)or CCD (charge-coupled-devices) image sensors, or on the software level. For example, in 2x2 binning, an array of 4 pixels becomes a single larger pixel, reducing the number of pixels 1/4. So it's a special form of downsampling.

- Applications
  - Deep-Sky Imaging: Enhances the detection of faint celestial objects by improving SNR.
  - Wide-Field Surveys: Balances resolution and sensitivity for large area coverage.

## Methods and Their Pros & Cons

- `cv::INTER_NEAREST`: nearest neighbor interpolation. This assigns the value of the nearest pixel to the pixel in the resized image. However, in high frequency areas (e.g., lots of edges), we might introduce aliasing due to not meeting the Nyquist Condition.
- `cv::INTER_AREA`: Calculate the **average** of nxn pixel blocks
- `cv::INTER_LINEAR`: Bilinear Interpolation: doing linear interpolation along X and Y directions, hence "bilinear" interpoliation. To explain the process, consider the example where we want to find the pixel value of `(2.2, 3.4)` on the original image.
    1. Choose image patch `I(2,3)=11`, `I(2,4)=12`, `I=(3,3)=13`, `I(3,4)=14`. `I()` means pixel value at each point
    2. Linear interpolation along X axos at `y=3`: `x = 11 + (2.2-2) / (3-2) * (13 - 11) = 11.4`
    3. Linear interpolation along X axos at `y=4`: `x = 12 + (2.2-2) / (3-2) * (14 - 12) = 12.4`
    4. Linear interpolation along Y axos at `x=2.3`: `y = 11.4 + (3.4 - 3) / (4-3) * (12.4 - 11.4) = 12.2`
- `cv::INTER_CUBIC`: Bicubic Interpolation (1981) [1] In OpenCV, interpolation is actually done in a simpler way than [the spline method](../2017/2017-01-26-interpolation.markdown). It skips estimating spline parameters, and use a standard bicubic kernel on the pixels directly. With `t=-1, 0, 1, 2`, the bicubic kernel function is ([reference](https://github.com/rootpine/Bicubic-interpolation/blob/master/bicubic.py)):

$$
\begin{cases}
(a + 2)|t|^3 - (a + 3)|t|^2 + 1, & \text{if } |t| \leq 1 \\
a|t|^3 - 5a|t|^2 + 8a|t| - 4a, & \text{if } 1 < |t| < 2 \\
0, & \text{if } |t| \geq 2
\end{cases}
$$

Here, f means the values of pixels. x1, x2, x3, x4 are the distance of x direction from new pixel to near 16 pixels. y1, ... are the distance of y direction. Then apply the kernel:

$$
dst(x, y) =
\begin{pmatrix}
u(x_1) & u(x_2) & u(x_3) & u(x_4)
\end{pmatrix}
\begin{pmatrix}
f_{11} & f_{12} & f_{13} & f_{14} \\
f_{21} & f_{22} & f_{23} & f_{24} \\
f_{31} & f_{32} & f_{33} & f_{34} \\
f_{41} & f_{42} & f_{43} & f_{44}
\end{pmatrix}
\begin{pmatrix}
u(y_1) \\
u(y_2) \\
u(y_3) \\
u(y_4)
\end{pmatrix}
$$

- `INTER_LANCZOS4`: Lanczos interpolation: sinusoid (1988) [2]

**So here is a short summary:**

| Method | Pros | Cons |
| ------ | ---- | ---- |
| Nearest neighbor | Fast | Jagged quality, aliasing |
| Inter-area | Smooth, no jaggedness | Not suitable for image upscaling |
| Bilinear Interlation | Decent quality, good for general purpose uses | Slight blurring, fewer sharp edges |
| Bicubic | Better quality | A bit slower than bilinear, might have ringing effect |
| Lanczos | Best quality, sharpness, minimal aliasing, used in FFmpeg| Slowest, potential ringing |

### 2D Cubic Interpolation (Catmull–Rom):

1. Find a row neighborhood 1x4 around the given point `(x,y)`
2. Given the intensity values at those 4 points, and a fraction $t$:

$$
\begin{gather*}
\begin{aligned}
& f(t) = 0.5 \left( 
    2p_1 +
    (-p_0 + p_2)t +
    (2p_0 - 5p_1 + 4p_2 - p_3)t^2 +
    (-p_0 + 3p_1 - 3p_2 + p_3)t^3
\right)
\end{aligned}
\end{gather*}
$$

3. Repeat 1 and 2 for 4 neighboring rows. 
4. Interpolate them along y axis:

$$
\begin{gather*}
\begin{aligned}
& I(x, y) = 0.5 \left( 
    2g_0 +
    (-g_{-1} + g_1)s +
    (2g_{-1} - 5g_0 + 4g_1 - g_2)s^2 +
    (-g_{-1} + 3g_0 - 3g_1 + g_2)s^3
\right)
\end{aligned}
\end{gather*}
$$

For the 1D case, as t changes, the interpolated value really depicts the transition of weights on each of the 4 points.

- While t=0, it's $p_1$. while t=1, it's $p_2$
- Derivative at `t=0` is $f'(0) = 0.5(p_2 - p_0)$


## Ring Effect

Lanczos filter might introduce ringing artifacts.

Without ringing
![Ringing_artifact_example_-_original](https://github.com/user-attachments/assets/0ea4ebc9-fac5-40db-9651-6d5b0bdedd54)

With ringing:
![Ringing_artifact_example](https://github.com/user-attachments/assets/c094954a-2511-48b4-9f6c-cd87031ffae6)

## Reference

[1] R. Keys, "Cubic convolution interpolation for digital image processing," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 29, no. 6, pp. 1153-1160, December 1981. URL: <http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1163711&isnumber=26156>

[2]  Lanczos, Cornelius (1988). Applied analysis. New York: Dover Publications. pp. 219–221. ISBN 0-486-65656-X. OCLC 17650089.
