---
layout: post
title: Python - Numpy and Pandas Operations
date: '2019-03-03 13:19'
subtitle: A Mumbo Jumbo Of Numpy and Pandas Operations and Examples
comments: true
header-img: "img/post-bg-2015.jpg"
tags:
    - Python
---

## Numpy

### General Programming Rules In Numpy

1. `np` allocates memory like C, so it's contiguous.
2. Math operations can be applied to all elements, which is a lot faster than for loop and use math module. Some examples include:
    - +,-, * /
    - `np.sqrt()`, `np.sin()`, `np.cos()`
    - Broadcast.

### Mathematical Matrix Operations

#### `np.max(array-like)`, `np.min(array-like)`

- Find the max / min in an array-like object

```python
# see 1
np.min((1,2))
```

#### Sum

- `np.sum(arr)` takes in array of booleans, ints, floats.

#### Average

- `np.mean(arr, axis=0)` might be slightly faster, and always returns the single mean value along the specified axis
- `np.average(arr, axis=0, weights=None, returned=False)` calculates the weighted average along the specified axis
  - If `weights` is None, `np.average` is pretty much the same as `np.mean`. Otherwise, one can specify the weights of each item along the specified axis
  - `Returned=True` when we want to return a tuple `(avg, summed_weights)`

```python
import numpy as np
a = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
print(np.mean(a, axis=1))  # See array([3., 3.])
print(np.average(a, axis=1))   # Also see array([3., 3.])

weights = [0.1, 0.1, 0.1, 0.2, 0.5]
avg, summed_weights = np.average(a, axis=1, weights=weights, returned=True)
print(f'Weighted average: {avg}, sumed_weights: {summed_weights}')
```

#### Comparisons

- `np.allclose(arr2, arr1)` returns `True` or `False`
- `np.isclose(arr2, arr1)` returns an array of `[True, False, ...]`

#### Padding

- `np.pad(array, pad_width, mode='constant', constant_values=(4, 6))` pads an array with:
  - `pad_width` is `[before, after]`, the constants in (4,6) before and after the array in the given axis

```python
import numpy as np
a = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
np.pad(a, ((2, 3), (1,2)), 'constant', constant_values=(4, 6))
```

The result is:

```python
array([[4, 4, 4, 4, 4, 4, 6, 6],
       [4, 4, 4, 4, 4, 4, 6, 6],
       [4, 1, 2, 3, 4, 5, 6, 6],
       [4, 1, 2, 3, 4, 5, 6, 6],
       [4, 6, 6, 6, 6, 6, 6, 6],
       [4, 6, 6, 6, 6, 6, 6, 6],
       [4, 6, 6, 6, 6, 6, 6, 6]])
```

See how along axis = 0 (rows) we have 4 before the array's row and 6 after, then along axis=1 (columns) we have 4 before the existing columns and 6 after?

#### Clipping

```Python
# gradients -- a dictionary containing the np.arrays
gradients = copy.deepcopy(gradients)

dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']

for gradient in [dWaa, dWax, dWya, db, dby]:
    np.clip(gradient, -maxValue, maxValue, out=gradient)
```

- Each `gradient` is an `np.array` object, so we are actually modifying gradient in place!
- If `gradient` is not an array, modifying in place won't make sense

### Random Number Generation

- `np.random.choice(a, size, replace=True, p=None)` randomly draws values from input `a`.

  - `a` is a list of numbers to draw from
  - `size` is the number of samples to draw
  - `replace` is if the same number can be drawn multiple times
  - `p` is the probability of each class. Uniform distribution is the default.

```python
arr = [1, 2, 3, 4, 5]
result = np.random.choice(arr, size=3)
# see [3,1,3]
```

### Non-Mathematical Matrix Operations

#### `np.where(pred)`

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

#### `numpy.ravel()`

- Takes in a multi-dimensional array and returns its content inside 1D array

```python
arr = np.array([[1,2],[3,4]])
print("arr")
print(arr)
print("arr.ravel()")
print(arr.ravel())

# see [1, 2, 3, 4]
```

#### ```np.unravel_index(indices, shape)```

- Convert `indices` of a linear array, into those in an n-d array with `shape`

```python
# index 6 in a linear array = (1,2) in a (3,4) array
np.unravel_index(6, (3, 4))

# index of 2 in a linear array = (0,2) in a (3,4) array
# index of 3 in a linear array = (0,3) in a (3,4) array
np.unravel_index([2,3], (3,4))
(array([0, 0]), array([2, 3]))
```

#### `np.meshgrid`

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

#### `np.array.squeeze()`

- remove axes with length 1. Example

```python
arr = np.array([[1,2,3]])
arr.squeeze() # np.array([1,2,3])
arr.squeeze()   # still sees np.array([1,2,3]) as no axis is of length 1.
```

#### `np.array.reshape (new_row, new_cln)`

- `np.array.reshape (new_row, new_cln)` is a common reshape function.

- `concat = np.concatenate((a_prev, xt), axis=0 )` concatenate two arrays together

#### `np.argmax(arr, axis)`

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

#### `np.in1d`

- Check if elements in a list has appeared in a list

```python
ls=["a", "b", "c"]
np.in1d(["a", "z", "f"], ls)
```

## Misc Function

- Input images

```python
if encoding == 'rgb8':
    image = np.frombuffer(data, np.uint8).reshape((height, width, 3))
```

- Before `[:, :, ::-1]`, the image channels are `[R, G, B]`
- After `[:, :, ::-1]`, the iamge channels become `[B, G, R]`

## Pandas

### Example Of Preparing Data For Training

```python
import os
import numpy as np
import pandas as pd

data = pd.read_csv("./bank-additional/bank-additional-full.csv", sep=";")
# trucated columns set to 500 so we can see all columns
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 50)

data["no_previous_contact"] = np.where(data["pdays"] == 999, 1, 0)
data["not_working"] = np.where(
    np.in1d(data["job"], ["student", "retired", "unemployed"]), 1, 0
)

model_data = pd.get_dummies(data)
model_data = model_data.drop([ "y_no", ], axis=1,)

train_data, test_data = np.split(
    model_data.sample(frac=1, random_state=1729), [int(0.9 * len(model_data))]
)
train_x = train_data.iloc[:, :-1]
train_y = train_data.iloc[:, 59]

test_x = test_data.iloc[:, :-1]
test_y = test_data.iloc[:, 59]

from sklearn.model_selection import train_test_split

X, val_X, y, val_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=2022, stratify=train_y
)
```

#### `model_data=pd.get_dummies(data)`

- Converts categorical variables into one-hot encoded variables. So if our data frame looks like:

| job        | marital   | education |
|------------|-----------|-----------|
| student    | single    | primary   |
| retired    | married   | secondary |
| unemployed | divorced  | tertiary  |

to:

| job_student | job_retired | job_unemployed | marital_single | marital_married | marital_divorced | education_primary | education_secondary | education_tertiary |
|-------------|-------------|----------------|----------------|-----------------|------------------|-------------------|---------------------|--------------------|
| 1           | 0           | 0              | 1              | 0               | 0                | 1                 | 0                   | 0                  |
| 0           | 1           | 0              | 0              | 1               | 0                | 0                 | 1                   | 0                  |
| 0           | 0           | 1              | 0              | 0               | 1                | 0                 | 0                   | 1                  |

#### `model_data.drop(["y_no"], axis=1)`

- Drops columns "y_no". `axis=1` indicates that it's the columns that we are interested in.

#### `df.iloc[row, column]`

- Selects elements in a dataframe by its integer location: `df.iloc[1]` selects the first row, `df.iloc[:, 1]` selects the first column
