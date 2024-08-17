---
layout: post
title: Deep Learning - TensorFlow Basics
date: '2022-01-26 13:19'
subtitle: Nothing Fancy, Just A Basic TF Network
comments: true
header-img: "img/home-bg-art.jpg"
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Basic Neural Net

The core features of TensorFlow (and many other Deep Learning Frames like PyTorch) are: 

- a cost function to calculate the model's total loss on the given inputs
- a computational graph to calculate gradients 
- an optimizer that applies gradient descent and other assistive optimization techniques to find the local minima in weights.

```python
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import time

tf.__version__
train_dataset = h5py.File('datasets/train_signs.h5', "r")
test_dataset = h5py.File('datasets/test_signs.h5', "r")

x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])
```

- `GradientTape` is a context manager that records operations?
- `tf.Tensor` is a tensor, an equivalent to numpy array with information for the computational graph.
- `tf.Variable(dtype)` it's best to specify the datatype here!

## Misc

- `HDF5` is "Hierarchical Data Format 5", a data format designed for compressing, chuking, and storing complex data hierarchies. It's similar to a filesystem, for example, you can create "groups" within an HDF5 file like creating a folder. Datasets are similar to files. HDF5 allows **access from multiple processes**, and is supported by multiple languages, like C, C++, Python. 

- ``