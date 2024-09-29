---
layout: post
title: Math - Transpose Convolution
date: '2017-01-07 13:19'
subtitle: Foundation For U-Net
comments: true
tags:
    - Math
---

## Example and Definition

### 1D Example

Given:
- Input: $[2,3]$
- Kernel: $[1,2,3]$
- Stride: 2

Solution:
Output Length is $(L_{input} - 1) \times stride + L_{kernel}$, which is $(2-1) \times 2 + 3 = 5$

so we can create an output matrix:

$$
\begin{bmatrix}
0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

So the first 3 items are: $2 \times [1,2,3] = [2,4,6]$ So the output matrix becomes:

$$
\begin{gather*}

\begin{bmatrix}
2 & 4 & 6 & 0 & 0
\end{bmatrix}
\end{gather*}
$$

In the output matrix, moving to the right by `stride=2`, the next 3 items are: $3 \times [1,2,3] = [3,6,9]$. Adding to the output, the output becomes:

$$
\begin{gather*}
\begin{bmatrix}
2 & 4 & 9 & 6 & 9
\end{bmatrix}
\end{gather*}
$$

### 2D Example

Given:

- Input

$$
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
$$

- Kernel:

$$
\begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
$$

- Stride: 2

Solution:

Output Height: $H = Stride \times (H_{input}-1) + H_{kernel} = 2 \times (2-1) + 2 = 4$
Output Width: $W = Stride \times (W_{input}-1) + W_{kernel} = 2 \times (2-1) + 2 = 4$

So let's create an output matrix:

$$
\begin{bmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{bmatrix}
$$

So starting in $output[0][0]$, we add $Input[0][0] \times kernel$:

$$
\begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
$$

In output, moving to the right by $stride=2$, we add $Input[0][1] \times kernel$:

$$
\begin{bmatrix}
2 & 0 \\
0 & -2
\end{bmatrix}
$$

...

Eventually, the output becomes

$$
\begin{bmatrix}
1 & 0 & 2 & 0 & \\
0 & -1 & 0 & -2 & \\
3 & 0 & 4 & 0 & \\
0 & -3 & 0 & -4 &
\end{bmatrix}
$$

