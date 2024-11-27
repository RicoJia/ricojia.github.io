---
layout: post
title: Python - Numpy View and Contiguous Array
date: '2019-03-06 13:19'
subtitle: View
comments: true
header-img: "img/post-bg-2015.jpg"
tags:
    - Python
---

## What is `np.view`

When slicing or transpose takes place on an array, we actually get a `np.view` object that shares the underlying memory with the array.

```python
import numpy as np
input_image = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])

# This is slicing
reverse_arr = input_image[:,::-1]
# True
print(f'Are reverse_arr and input_image sharing memory? {np.may_share_memory(input_image, reverse_arr)}')

# False
con_reverse_arr = np.ascontiguousarray(reverse_arr)
print(f'Are con_reverse_arr and input_image sharing memory? {np.may_share_memory(input_image, con_reverse_arr)}')

# This is transpose
transposed_arr = input_image.T
# True
print(f'Are con_reverse_arr and input_image sharing memory? {np.may_share_memory(input_image, transposed_arr)}')
```

- `np.may_share_memory()` checks if two arrays share the same underlying memory

### How does it work?

An `np.array` has 4 important compoenents:

- `data` buffer
- `shape`
- `strides`
- `dtype`

In the above array, we have

```
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
Strides: (32, 8) 
Shape: (3, 4) 
Dtype: int64 
```

The databuffer is 1D array.

- `strides` means that to go to the next row, on the 1D array, one needs to jump `32 bytes (32/8=4 int)`.
- To go to the next column, one needs to jump `8/8 = 1 int`.

On the transpose array:

```
[[ 1  5  9]
 [ 2  6 10]
 [ 3  7 11]
 [ 4  8 12]]
Strides: (8, 32) 
Shape: (4, 3) 
Dtype: int64 
```

- To go to the next row: one needs to jump 1 int. To go to the next column, one needs to jump 4 int.

## When Is `np.ascontiguous` Necessary

When interfacing with other libraries, like `torch`, `OpenCV`, or a low level C library, arrays are often expected to be contiguous. When we have an array that's actually a view object, we can use `np.ascontiguous()` to copy the array into a new array.

```python
import numpy as np
input_image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# This is slicing
reverse_arr = input_image[:,::-1]
print(reverse_arr.flags['C_CONTIGUOUS'])  # Output: False
# True
print(f'Are reverse_arr and input_image sharing memory? {np.may_share_memory(input_image, reverse_arr)}')

con_reverse_arr = np.ascontiguousarray(reverse_arr)
print(con_reverse_arr.flags['C_CONTIGUOUS'])  # Output: True
# False
print(f'Are con_reverse_arr and input_image sharing memory? {np.may_share_memory(input_image, con_reverse_arr)}')
```
