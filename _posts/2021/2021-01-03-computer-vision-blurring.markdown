---
layout: post
title: Computer Vision - Blurring
date: '2021-01-03 13:19'
subtitle: Gaussian Blurring (Under Active Updates)
comments: true
header-img: "home/bg-o.jpg"
tags:
    - Computer Vision
---

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
