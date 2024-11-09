---
layout: post
title: Math - Different Types Of Convolutions
date: '2017-01-07 13:19'
subtitle: Transpose Convolution (U-Net), Dilated Convolution (DeepLab)
comments: true
tags:
    - Math
---

# Transpose Convolution

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

**Padding controls how much space is added around the output.** It is the padded area around the intermediate output, so we just need to carve it out and get the output

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

## Remarks

- What does "same" padding mean in TensorFlow's transpose_convolution? Answer: 'same' would yield `output_size = stride * input_size`. Please [see here](https://community.deeplearning.ai/t/week4-question-to-the-padding-of-conv2dtranspose/25331/3?u=ricoruotongjia).
    - A common mistake in using TensorFlow Convolution or transpose_convolution is: **accidentally putting channels at the wrong dimension.** E.g., Why would I get (1,2,4,1) if my input is (1,1,2,2) for transpose_convolution? Because by default, `data_format = "channels_last"`. If your input is (1,1,2,2) with 1 output filter channel, you will get (1,2,4,1).

# Dilated Convolution

The word "à trous" in French means "hole". The A Trous algorithm was conventionally used in wavelet transform, but now it's in convolutions for deep learning.

Each element in the dilated conv kernel now takes a value in a subregion. This can effectively increase the area the kernel sees, since in most pictures, a small subregion's pixels are likely similar

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/e14d5e18-6eb2-4cff-9d03-97ad9240988e" height="300" alt=""/>
       </figure>
    </p>
</div>

Along 1D, dilated convolution is:

$$
\begin{gather*}
y[i] = \sum_K x[i + rk] w[k]
\end{gather*}
$$

Where **the rate** `r=1` is the regular convolution case. We can have padding like the usual convolution as well.

**Why is Dilated (Atrous) Convolution useful?** Because it can increase the receptive field of a layer in the input. DeepLab V1 & V2 uses ResNet-101 / VGG-16. The original networks' last pooling layer (pool5) or convolution 5_1 is to 1 to avoid too much signal loss. But DeepLab V1 and V2 uses atrous convolution in all subsequent layers using a `rate=2`


Intro TODO

## Padding Calculation

Effectively, we have a larger kernel and a larger stride. With input image size `n`, kernel size `k`, stride `s`, padding (conventionally one side only)`p`, dilated rate `r`, **output size o** is:

- The effective kernel size `g` is: $g = r x (k - 1) + 1$,
- The effective stride is still `s`
- So $o = \frac{n + 2*p - g}{s} + 1$. To keep `input_dim = output_dim`, i.e., `n=o` (same padding):

$$
\begin{gather*}
p = \frac{s(o-1) + g - o}{2}
\end{gather*}
$$

- So in the special case where stride is 1: 

$$
\begin{gather*}
p = \frac{g - 1}{2}
\end{gather*}
$$

However, we commonly use `3x3` kernels, whose effective kernel size is an odd number. So, we have to do:

```python
def fixed_padding(kernel_size, dilation):
    # From mobilenet v2
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return (pad_beg, pad_end, pad_beg, pad_end)
```

