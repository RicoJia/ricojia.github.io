---
layout: post
title: Computer Vision - Blurring
date: '2021-01-03 13:19'
subtitle: Box Blur, Gaussian Blurring (Under Active Updates)
comments: true
header-img: "home/bg-o.jpg"
tags:
    - Computer Vision
---

## Box Blur

A box blur is basically an average filter: for every pixel, average its neighborhood, like a 3x3 window around it. It can be implemented as a cross-correlation. One important detail is that the box blur **centers its kernel on the output pixel**. In 1D, with a 3-pixel kernel:

$$
\begin{gather*}
v[i]=\frac{x[i-1]+x[i]+x[i+1]}{3}
\end{gather*}
$$

The kernel is symmetric, so cross-correlation and convolution give the same result here.

**Example.** With zero padding at the borders:

$$
\begin{gather*}
x=[0,0,9,0,0],\qquad B=\frac13[1,1,1] \\
\boxed{v=[0,3,3,3,0]}
\end{gather*}
$$

### Repeated Box Blurs Look Gaussian

If we apply the box blur twice, the output starts to look Gaussian-blurred:

$$
\begin{gather*}
\begin{aligned}
w[0]&=(0+0+3)/3=1\\
w[1]&=(0+3+3)/3=2\\
w[2]&=(3+3+3)/3=3\\
w[3]&=(3+3+0)/3=2\\
w[4]&=(3+0+0)/3=1
\end{aligned} \\
\boxed{w=[1,2,3,2,1]}
\end{gather*}
$$

The more passes we apply, the closer it gets to a Gaussian. This is because convolution is associative, so `n` passes of `B` are the same as one pass of `B` convolved with itself `n` times:

$$
\begin{gather*}
(x*B)*B=x*(B*B) \\
x_n=x*\underbrace{(B*B*\cdots*B)}_{n\text{ times}}
\end{gather*}
$$

Indeed, `B*B = [1, 2, 3, 2, 1] / 9`, which is exactly the shape of `w` above.

A box kernel can be interpreted as a uniform probability distribution. Convolving it repeatedly corresponds to adding independent, uniformly distributed random offsets, so by the [Central Limit Theorem](https://ricojia.github.io/2017/02/04/stats-basic-recap/#central-limit-theorem) the combined kernel approaches a Gaussian. See [Repeated Box Filters Approach A Gaussian](https://ricojia.github.io/2017/02/04/stats-basic-recap/#repeated-box-filters-approach-a-gaussian) for its variance.

## Gaussian Blurring

TODO

### OpenCV Implemenation 

```python
def get_gaussian_kernel(size:int): 
    """
    We're not sure the padding rule of cv2.GaussianBlur - Technically applying the identity image with 
    the same kernel size should yield the Kernel, But it doesn't
    """
    #identity image here
    image = np.zeros((size+2, size+2))
    image[int(size/2) + 1, int(size/2) + 1] = 1
    kernel = cv2.GaussianBlur(image, (size, size), 0)[1:1+size, 1:1+size]
    return kernel
```

- Kernel size needs to be **odd**
