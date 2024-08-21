---
layout: post
title: Python - Numpy Operations Mumbo Jumbo
date: '2019-03-03 13:19'
subtitle: A Mumbo Jumbo Of Python Operations
comments: true
header-img: "img/home-bg-2015.jpg"
tags:
    - Python
---

## General Programming Rules In Numpy

1. `np` allocates memory like C, so it's contiguous.
2. Math operations can be applied to all elements, which is a lot faster than for loop and use math module. Some examples include:
    - +,-, * /
    - `np.sqrt()`, `np.sin()`, `np.cos()`
    - Broadcast.

## Mathematical Matrix Operations

### `np.max(array-like)`, `np.min(array-like)`

- Find the max / min in an array-like object

```python
# see 1
np.min((1,2))
```

## Non-Mathematical Matrix Operations

### `np.where(pred)`

- Returns a tuple where `ith` element represents the indices along `ith` axis that satisfies the pred

```python
array = np.array([[1, 6, 3], 
                  [7, 2, 8], 
                  [4, 9, 5]])
# See (x_indices, y_indices) where array>5 is true
# (array([0, 1, 1, 2]), array([1, 0, 2, 1]))
indices = np.where(array > 5)
```

- Conditional Assignment: `res=np.where(cond, array1, array2)`. when `cond=True`, `res` gets `array1` value, otherwise gets `array2` value

```python
array = np.array([10, 5, 8, 3, 12])
# See array([10,  0,  8,  0, 12])
res = np.where(array > 5, array, 0)
```

### ```np.unravel_index(indices, shape)```

- Convert `indices` of a linear array, into those in an n-d array with `shape`

```python
# index 6 in a linear array = (1,2) in a (3,4) array
np.unravel_index(6, (3, 4))

# index of 2 in a linear array = (0,2) in a (3,4) array
# index of 3 in a linear array = (0,3) in a (3,4) array
np.unravel_index([2,3], (3,4))
(array([0, 0]), array([2, 3]))
```

### `np.meshgrid`

- Generate coordinate matrices (grids) from coordinate vectors. It's often used in plotting functions, i.e., when you need to evaluate functions on a grid. 

```python
# See two "grid" generated. The result is [2,5,3]

# array([[[0, 0, 0],
#         [1, 1, 1],
#         [2, 2, 2],
#         [3, 3, 3],
#         [4, 4, 4]],
#
#        [[0, 1, 2],
#         [0, 1, 2],
#         [0, 1, 2],
#         [0, 1, 2],
#         [0, 1, 2]]])
res = np.mgrid[0:5, 0:3]
```

### `np.array.squeeze()`

- remove axes with length 1. Example

```python
arr = np.array([[1,2,3]])
arr.squeeze() # np.array([1,2,3])
arr.squeeze()   # still sees np.array([1,2,3]) as no axis is of length 1.
```

### `np.array.reshape (new_row, new_cln)`

- `np.array.reshape (new_row, new_cln)` is a common reshape function. 

### `np.argmax(arr, axis)`

- Finding the args of max of an array along an axis

```python
import numpy as np
y = np.array([0, 2, 1, 3])
# one hot vector
one_hot = np.array([
    [1, 0, 0, 0],  # Corresponds to class 0
    [0, 0, 1, 0],  # Corresponds to class 2
    [0, 1, 0, 0],  # Corresponds to class 1
    [0, 0, 0, 1],  # Corresponds to class 3
])
np.argmax(one_hot, axis=1)
```
