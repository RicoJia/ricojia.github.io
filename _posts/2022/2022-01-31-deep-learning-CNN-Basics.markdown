---
layout: post
title: Deep Learning - CNN Basics
date: '2022-01-31 13:19'
subtitle: Filters, Padding, Convolution and Its Back Propagation, Receptive Field
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Filters

Filters (aka kernels): "Pattern Detectors". Each filter is a small matrix, which you can drag along an image and multiply pixel values with (convolution). They can detect edges, corners, and later, parts of dogs.

When a filter cross correlates with a matrix that has the same shape, it will generate a high response. If it encounters an "opposite" shape, it will generate a negative response.

Below is an 45 deg edge filter

$$
\begin{gather*}
\begin{bmatrix}
-1 & -1 & 2 \\
-1 & 2 & 1 \\
2 & 1 & 1
\end{bmatrix}
\end{gather*}
$$

### Cross Correlation vs Convolution

In signal processing, a filter convolves with an image by reversing the filter horizontally and vertically.
    $$
    \begin{bmatrix}
    1 & 2\\
    3 & 4
    \end{bmatrix}
    \\ =>
    \begin{bmatrix}
    4 & 3\\
    2 & 1
    \end{bmatrix}
    $$

Why?

- In 1D signal processing, a **causal system** could have an impulse response [1,2,3]. If we now send an impulse, which is `[1,0,0]` at `[t1, t2, t3]`, we expect to see [1,2,3], but to represent that in a sliding window "[0, 0, 1]"  (**flipped in time**) so at t1 we get 1, t2 we get 2, t3 we get 3. This is the causality consistency, or "the correctness of sequence of outputs"
- For feature detection, we don't need to flip the kernel. Consider this scenario: we want to detect a 1D slope `[1,0]`. Say we have a slope: `[3,2,1,0]`. If we flip the kernel, we will get **negative responses**! So, no need to flip the kernel for CNN. In the machine learning community, **we apply "cross-correlation" but we often call it "convolution".**

## Padding

Why padding?

- **Keeps info at the borders**. Information at the borders will be washed away too quickly
- **Allows us to design deeper neural networks**, Otherwise the inputs will diminish

With image size `n`, kernel size "f", stride `s`, padding (conventionally one side only)`p`, **the padding formula** is:

$$
\begin{gather*}
\frac{n + 2*p - f}{s} + 1
\end{gather*}
$$

There are three types of padding:

- 'valid' : no padding at all.

```
[1, 2, 3]
[1, 1, 1]
=> [6]
```

- 'same': output is the same as the input

```
[0, 1, 2, 3, 0]
[1, 1, 1]
    [1, 1, 1]
        [1, 1, 1]
=> [3, 6, 5]
```

- 'full', Pads even more zeros, so each pixel in input touches the kernel same number of times

```
[0, 0, 1, 2, 3, 0, 0]
[1, 1, 1]
    [1, 1, 1]
        [1, 1, 1]
            [1, 1, 1]
            [1, 1, 1]
=> [1, 3, 6, 5, 3]
```

## Pooling

Pooling is a form of downsampling. There are average pooling (less commonly used) and max pooling (more commonly used)

Max pooling is to get the max within a window to retain the most salient feature. Then, move the kernel to the right and to the left by stride after 1 operation. Average pooling is instead of taking the max within a window, take the average.

**Global max pooling** is to condense a feature map into a single value, so it can be used to detect if a feature is present or not. This helps reduce overfitting.

### Example

Given an image:

```
1  3  2  4
5  6  7  8
9  2  3  1
4  0  1  5
```

If the window size is 2x2, with stride = 2, we get

```
6, 8,
9, 5
```

## Convolutional Layer

The Convolutional Neural Network (CNN) was invented by Yann Lecun in 1990. Each convolution layer has multiple filters.

### Example CNN Layer

The input has 3 channels (e.g., RGB), the output has 4 channels? Each output channel is **a feature map**, or **people call it "filters"**. So if we need 3 kernels for each output channel, there'd be in total 3 x 4 =12 kernels.

<p align="center">
<img src="https://user-images.githubusercontent.com/77752418/168213042-c11b63e7-0207-4fc3-ba41-e804dce06107.gif" height="400" width="width"/>
</p>

What the illustration doesn't show is **bias** addition: `output[m, o] += bias[o]` bias is added across output channels. For each output channel, **bias is a single number.**

### Example CNN Network

Example: say we have an 32x32x1 grayscale image:

1. First layer is a convolutional layer with 3x3x2 filters with stride = 1 with no padding.
    1. The direct convolution output is 30x30x2.
    2. Then the output goes through an activation function
    3. Then the output goes through a pooling layer with a 2x2 window size, stride =1, and finally outputs 15x15x2

2. Second layer is a convolutional layer with 4 3x3x2 filters with stride = 1 with padding of 2 (so input and output layer size are the same).
    1. Each filter takes in both channel, then convolve with the two input channels, add them, then outputs. So we will get 15x15x4 -> activation function -> pooling layer.
3. The third Flattening layer: 15x15x4 -> 900 x1 vector. Note that all channels are flattened into one
4. Fully connected layer

## Theories

### How does back propagation on a kernel `K` work?

Below was inspired by the  [Youtube channel far1din](https://www.youtube.com/watch?v=z9hJzduHToc)

One Neural Net implementation [can be found here](https://github.com/TheIndependentCode/Neural-Network)

- Given input X, kernel K, if we expand an element in the output Y:

    $$
    \begin{gather*}
    y_{11} = x_{11}k_{11} + x_{12}k_{12} + x_{13}k_{13} + ... 
    \\
    y_{12} = x_{12}k_{11} + x_{13}k_{12} + x_{14}k_{13} + ...
    \\
    y_{21} = x_{21}k_{11} + x_{22}k_{12} + x_{23}k_{13} + ...
    \end{gather*}
    $$

- We need to get $\frac{J}{W}$ and $\frac{J}{X}$.

    $$
    \begin{gather*}
    \frac{\partial J}{\partial x_{11}} = \frac{\partial J}{\partial y_{11}}k_{11}
    \\
    \frac{\partial J}{\partial x_{12}} = \frac{\partial J}{\partial y_{11}}k_{12} + \frac{\partial J}{\partial y_{12}}k_{11}
    \end{gather*}
    $$

- For kernel gradient $\frac{\partial J}{\partial K}$, it's cross correlation: $x \ast \frac{\partial J}{\partial y}$

    $$
    \begin{gather*}
    \frac{\partial J}{\partial k_{11}} = \frac{\partial J}{\partial y_{11}}x_{11} + \frac{\partial J}{\partial y_{12}}x_{12} + \frac{\partial J}{\partial y_{21}}x_{21} + \frac{\partial J}{\partial y_{22}}x_{22}
    \\
    \frac{\partial J}{\partial k_{12}} = \frac{\partial J}{\partial y_{11}}x_{12} + \frac{\partial J}{\partial y_{12}}x_{13} + \frac{\partial J}{\partial y_{21}}x_{22} + \frac{\partial J}{\partial y_{22}}x_{23}
    \end{gather*}
    $$

- For bias gradient, there is only 1 bias value per output channel; so, we apply it to all elements of one channel. Its gradient is the sum across all channels, so its output gradient $\sum_c \frac{\partial J}{\partial y_c}$

- For input gradient, $\frac{J}{X}$, it's actually convolution: $k \circledast \frac{\partial J}{\partial y}$
    $$
    \begin{gather*}
    \frac{J}{x_{11}} = \frac{J}{y_{11}}k_{11} 
    \\ \frac{J}{x_{12}} = \frac{J}{y_{11}}k_{12} + \frac{J}{y_{12}}k_{11}
    \\ \frac{J}{x_{13}} = \frac{J}{y_{11}}k_{13} + \frac{J}{y_{12}}k_{12} + \frac{J}{y_{12}}k_{11}
    ...
    \end{gather*}
    $$

    - See? This is convolution!

## Implementation

```python

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, padding=0) -> None:
        # n is the number of inputs, p is the number of outputs
        # nxn, [output_channels, input_channels, kernel, kernel]
        self.weights = he_init_cnn(
            out_channels=out_channels, in_channels=in_channels, kernel_size=kernel_size
        )
        self.bias = np.zeros(out_channels, dtype=np.float32)
        self.stride = 1
        self.kernel_size = np.asarray(kernel_size, dtype=np.float32)
        self.padding = padding

    def pad_input(self, x):
        if self.padding > 0:
            # Here (0,0) for the first axis as that's the batch dimension, then input channel,
            # then (self.padding, self.padding) for the rows, and columns
            return np.pad(
                x,
                (
                    (
                        (0, 0),
                        (0, 0),
                        (self.padding, self.padding),
                        (self.padding, self.padding),
                    )
                ),
                mode="constant",
            )
        return x

    def __call__(self, x):
        # Forward pass: input [batch_numer, input_channels, height, weight]
        out_channel_num, input_channel_num = (
            self.weights.shape[0],
            self.weights.shape[1],
        )
        batch_num = x.shape[0]
        if x.ndim == 3:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        input_image_size = np.asarray((x.shape[2], x.shape[3]), dtype=np.float32)
        output_size = (
            (input_image_size + self.padding * 2 - self.kernel_size) / self.stride + 1
        ).astype(int)
        self.output = np.zeros(
            [batch_num, out_channel_num, output_size[0], output_size[1]],
            dtype=np.float32
        )

        if x.shape[1] != input_channel_num:
            raise ValueError(
                f"Number of input channel must be {input_channel_num}, but now it is {x.shape[1]}"
            )
        x = self.pad_input(x).astype(np.float32)
        self.input = x
        for b in range(batch_num):
            for o in range(out_channel_num):
                for i in range(input_channel_num):
                    self.output[b, o] += scipy.signal.correlate2d(
                        x[b][i], self.weights[o][i], mode="valid"
                    ).astype(np.float32)
                self.output[b, o] += self.bias[o]
        return self.output

    def backward(self, output_gradient):
        if output_gradient.shape != self.output.shape:
            raise ValueError(
                f"Output Gradient Shape {output_gradient.shape} must be equal to output shape {self.output.shape}"
            )
        out_channel_num, input_channel_num = (
            self.weights.shape[0],
            self.weights.shape[1],
        )
        self.output_gradient = output_gradient.astype(np.float32)
        self.input_gradient = np.zeros(self.input.shape, dtype=np.float32)
        # in this case, weights is kernel
        self.weights_gradient = np.zeros(self.weights.shape, dtype=np.float32)  # delJ/delK

        batch_num = self.input.shape[0]
        for b in range(batch_num):
            for o in range(out_channel_num):
                for i in range(input_channel_num):
                    self.weights_gradient[o, i] = scipy.signal.correlate2d(
                        self.input[b, i], output_gradient[b, o], mode="valid"
                    ).astype(np.float32)
                    self.input_gradient[b, i] += scipy.signal.convolve2d(
                        output_gradient[b, o], self.weights[o][i], mode="full"
                    ).astype(np.float32)

        if self.padding > 0:
            # Just keep the unpadded portion, which is consistent with pytorch
            self.input_gradient = self.input_gradient[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]
        self.bias_gradient = np.sum(output_gradient, axis=(0, 2, 3)).astype(np.float32)
        return self.input_gradient

c = Conv2d()
c() # consistent with how it's called in PyTorch
c.backward()

```

### Benefits Of Using Convolutional Layer

- Reduces the number of parameters, thus reduces the overfitting problem.
- **Smaller number of parameters also means smaller sets of training images**
- Convolutional Layers also benefit from **sparsity of connections**. This means that the activation of the next layer is only affected by a small number of activations from the previous layer (the ones in the corresponding filtered area)

## Receptive Field

A receptive field of a feature of an element in a feature map refers to **the size of region of the input that produces this feature**. E.g., say we have 2 CNN layers:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/a69ffcac-a7b6-4203-9f95-1ebe430b8c9f" height="300" alt=""/>
        <figcaption><a href="https://www.researchgate.net/figure/The-receptive-field-of-each-convolution-layer-with-a-3-3-kernel-The-green-area-marks_fig4_316950618">Source: Research Gate </a></figcaption>
    </figure>
</p>
</div>

We denote the input as $f_0$, output of layer $l$ as $f_l$. The receptive field of layer $l$ is $r_l$, but really **it's the number of cells on layer $l$ that a reference layer output sees**. In the below example, $r_0=8$ at the input, w.r.t the second layer. It would be different w.r.t a different layer.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/6020572c-bbeb-46bf-b901-1b73f436fdd6" height="300" alt=""/>
        <figcaption><a href="https://www.baeldung.com/cs/cnn-receptive-field-size">Source</a></figcaption>
    </figure>
</p>
</div>

With the above in mind, from the reference layer output to the input layer, recursively:

$$
\begin{gather*}
r_{l-1} = s_lr_l + k_l-s_l
\end{gather*}
$$

For more formulas and an interactive representation, [please see Araujo's work](https://distill.pub/2019/computing-receptive-fields/).


