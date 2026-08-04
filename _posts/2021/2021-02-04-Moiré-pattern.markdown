---
layout: post
title: Computer Vision - Moiré Pattern
date: 2021-02-04 13:19
subtitle:
comments: true
tags:
  - Computer Vision
---
## Noise Patterns

| Pattern                 | Cause                                                       | How to recognize it                                        |
| ----------------------- | ----------------------------------------------------------- | ---------------------------------------------------------- |
| Moiré                   | Two similar spatial patterns or grids                       | Changes strongly with scale, angle, or resolution          |
| Aliasing                | Sampling is too coarse                                      | Changes when the sampling rate changes                     |
| Interference            | Coherent waves add constructively and destructively         | Depends strongly on phase and geometry                     |
| Speckle                 | Many small coherent returns arrive with random phases       | Grainy texture that changes after very small sensor motion |
| Multipath               | A signal reaches the sensor through several paths           | Creates ghosts, fading, smearing, or repeated targets      |
| Sidelobes               | A finite sensor array responds away from its main direction | False responses tied to the array and beam geometry        |
| Ringing                 | Filtering or limited bandwidth around a sharp edge          | Repeated ripples next to strong edges                      |
| Fixed-pattern noise     | Sensor channels have different gains or offsets             | Rows or columns stay fixed in sensor coordinates           |
| Rolling-shutter banding | Lighting frequency interacts with camera timing             | Bands change with exposure time or frame rate              |

## The main idea

A **moiré pattern** appears when two repeating patterns are almost—but not quite—the same.

Imagine placing two window screens on top of each other. Each screen has very fine lines. If one screen is rotated slightly, you see large dark and light bands that are not present in either screen alone. Those large bands are **the moiré pattern**.

![](https://www.futurity.org/wp/wp-content/uploads/2023/06/moire_1600.jpg)

The same thing can happen when: a camera sees fine fabric or mesh. When **the scene pattern is too fine for the pixel grid**, the camera cannot sample it correctly. A false, lower-frequency pattern appears. So this is **aliasing**. Moiré is one visible form of spatial aliasing:

> Aliasing is the sampling problem. Moiré is the large false pattern that may appear because of it.

This is why photographing a monitor often produces colored waves or bands. The monitor has its own pixel grid, and the camera has another pixel grid.

Changing any of the following can change or remove the pattern:

- camera distance;
- focus;
- image resolution;
- crop or resize ratio.

Anti-alias filtering works best before or during sampling. After a false frequency has been recorded, **later filtering may hide the pattern**, but it cannot always recover the lost detail. See 

## Moire Pattern Frequency

Suppose the two patterns have spatial frequencies $f_1$ and $f_2$. The moiré frequency is approximately their difference:

$$
f_{\text{moire}} = |f_1-f_2|.
$$

The distance between the large moiré bands is

$$
P_{\text{moire}}
=
\frac{1}{|f_1-f_2|}.
$$

For example, suppose one grid has 100 lines per metre and the other has 102:

$$
f_1=100\ \text{m}^{-1},
\qquad
f_2=102\ \text{m}^{-1}.
$$

Their difference is only

$$
f_{\text{moire}}=2\ \text{m}^{-1}.
$$

Therefore, the visible bands are

$$
P_{\text{moire}}=\frac{1}{2}=0.5\ \text{m}
$$

apart.

The original lines are about one centimetre apart, but their interaction creates bands half a metre apart.

