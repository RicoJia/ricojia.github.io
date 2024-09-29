---
layout: post
title: Math - Transpose Convolution
date: '2017-01-07 13:19'
subtitle: Foundation For U-Net
comments: true
tags:
    - Math
---

## Definition

Transpose Convolution (a.k.a upsampling convolution) was designed specifically for upsampling, which is then used in the decoder part of an encoder-decoder network for restructing layers. The intuition is that this **reverses a regular convolution operation**. Remember the regular convolution formula?

$$
output size = \frac{i + 2p - k}{s} + 1
$$

Where: `i` is the input size, `p` is padding, `k` is the kernel size, s is the stride.  So for transpose convolution,
$$
output size = (i - 1)s + k - 2p
$$

Below is an example of Transpose Convolution: `i=4, k=3, p=2, s=1`

![637bb302cd268d9b657fb478d39c150f](https://github.com/user-attachments/assets/5dabf1f9-967b-45d3-8ded-57f1755807dd)


## Illustrative Examples

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
1 & 1 & 1 \\
0 & 0 & 0 \\
-1 & -1 & -1
\end{bmatrix}
$$

- Stride: 2
- Padding: 1

Solution:

**Padding controls how much space is added around the output.** It is the padded area around the intermediate output, so we just need to carve it out and get the output**

Intermediate output Height: $H = Stride \times (H_{input}-1) + H_{kernel} - 2 \times padding = 2 \times (2-1) + 3 = 5$
Intermediate output Width: $W = Stride \times (W_{input}-1) + W_{kernel} - 2 \times padding = 2 \times (2-1) + 3 = 5$

Output Height: $H = Stride \times (H_{input}-1) + H_{kernel} - 2 \times padding = 2 \times (2-1) + 3 - 2 = 3$
Output Width: $W = Stride \times (W_{input}-1) + W_{kernel} - 2 \times padding = 2 \times (2-1) + 3 - 2 = 3$

So let's create an intermediate matrix:

$$
\begin{bmatrix}
0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

So starting in $output[0][0]$, we add $Input[0][0] \times kernel$:

$$
\begin{bmatrix}
1 & 1 & 1 \\
0 & 0 & 0 \\
-1 & -1 & -1
\end{bmatrix}
$$

In the intermediate output, moving to the right by $stride=2$, we **add** $Input[0][1] \times kernel$:

$$
\begin{bmatrix}
2 & 2 & 2 \\
0 & 0 & 0 \\
-2 & -2 & -2
\end{bmatrix}
$$

...

Eventually, the output becomes

$$
\begin{bmatrix}
1 & 1 & 3 & 2 & 2 \\
0 & 0 & 0 & 0 & 0 \\
2 & 2 & 4 & 2 & 2 \\
0 & 0 & 0 & 0 & 0 \\
-3 & -3 & -7 & -4 & -4
\end{bmatrix}
$$

Carve the padding out, we get

$$
\begin{bmatrix}
0 & 0 & 0 \\
2 & 4 & 2 \\
0 & 0 & 0 \\
\end{bmatrix}
$$

Code: 

```python
import torch
from torch import nn
 
# Input
Input = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
#Kernel
Kernel = torch.tensor([[1.0, 1.0, 1.0], [0,0,0], [-1,-1,-1]])
 
# # Redefine the shape in 4 dimension
Input = Input.reshape(1, 1, 2, 2)
Kernel = Kernel.reshape(1, 1, 3, 3)
 
# Transpose convolution Layer
Transpose = nn.ConvTranspose2d(in_channels =1, 
                               out_channels =1,
                               kernel_size=3, 
                               stride = 2, 
                               padding=1, 
                               bias=False)
 
Transpose.weight.data = Kernel
Transpose(Input)
```