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

## Basic Operations

### Max Operations
- Immutable (`tf.constant`) vs Variable (`tf.Variable`), notice the different capitalization:
- `tf.math.reduce_max()`: find the max along certain dimension(s).

```python
# immutable vs variable
import tensorflow as tf

var = tf.Variable([[1, 2], [3, 4]])
con = tf.constant([[1, 2], [3, 4]])

# Calculate the max along certain dimension(s). Doc says: 
# If axis is None, all dimensions are reduced, and a tensor with a single element is returned.
# By default, keepdims is False, so this will collapse the matrix
tf.math.reduce_max(
    var, axis=0, 
)   # see <tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 4], dtype=int32)>

tf.math.reduce_max(
    var
)   # see <tf.Tensor: shape=(2,), dtype=int32, numpy=4>

```

- [`tf.math.argmax`](https://www.tensorflow.org/api_docs/python/tf/math/argmax): returns the index with the largest value across axes of a tensor. **By default, it returns max along axis=0**

```python
var = tf.Variable([[2,1], [3, 4]])
tf.math.argmax(var, axis=-1)    # see <tf.Tensor: shape=(2,), dtype=int64, numpy=array([0, 1])>
tf.math.argmax(var) # <tf.Tensor: shape=(2,), dtype=int64, numpy=array([1, 1])>
```

### Reshaping

- Transpose: `tf.transpose(T)`
- Permutation: `A_t = tf.transpose(A, perm=[0, 2, 1])`
- Reshape: `tf.reshape(tensor, shape)`

```python
# 2x3
t = tf.constant([[1, 2, 3],
                 [4, 5, 6]])
# reshaping into 3x2
reshaped_tensor = tf.reshape(t, [3, 2])

batch_size, H, W = 3, 2, 3
tensor = tf.random.normal([batch_size, H, W])
tf.reshape(tensor, [-1, H*W])
```
   - `-1` will make tf infer the previous tensor sizes

- Squeeze: `tf.squeeze(tensor, dim)` to remove dims with only one element

```python
t = tf.constant([[[1, 2, 3],
                 [4, 5, 6]]])
# squeezing into 3x2
reshaped_tensor = tf.squeeze(t)
```

### Element-wise Operations

- Broadcasting: this is similar to numpy

```python
([0.9, 0.3, 0.4, 0.5, 0.1] < 0.4) # see [False, True, False, False, True]
```

- Masking: this is similar to numpy as well. Its numpy equivalent is `tensor[mask]`. Note that from the doc:

> `0 < dim(mask) = K <= Element-wisedim(tensor)`, and mask's shape must match the first K dimensions of tensor's shape. We then have: `boolean_mask(tensor, mask)[i, j1,...,jd] = tensor[i1,...,iK,j1,...,jd]` where (i1,...,iK) is the ith True entry of mask (row-major order). The axis could be used with mask to indicate the axis to mask from.

**So masking will collapse all matching dims into one**

```python
# 3x2
tensor = [[1, 2], [3, 4], [5, 6]]
# 3
mask = np.array([True, False, True])
# so the two's 1st dimension match! based on that, all the matching dimensions will be collapsed into 1, while the other dims will remain
tf.boolean_mask(tensor, mask)  # [[1, 2], [5, 6]]
```

- Element-wise multiplications: `*` and `tf.multiply()`

```python
tensor1 = tf.constant([0, 1, 2, 3])  # 1-D example
tensor2 = tf.constant([1, 2, 3, 4])  # 1-D example
tensor1 * tensor2
tf.multiply(tensor1, tensor2) # see <tf.Tensor: shape=(4,), dtype=int32, numpy=array([ 0,  2,  6, 12], dtype=int32)>
tensor1 .* tensor2  # Invalid
```

- Elementwise exponentiation:

```python
tensor ** 2
```

- Elementwise sum:

```python
tf.reduce_sum(tensor)

```

### Gradient

- `tf.GradientTape()`: record operations on tensors so we can calculate gradients more easily. `GradientTape` is a context manager that records operations

```python
import tensorflow as tf

# Example variables
x = tf.constant(3.0)
w = tf.Variable(2.0)

# Gradient computation using GradientTape
with tf.GradientTape() as g:
    # Watch the variable `w` (automatically watched since it's a Variable)
    y = w * x + 2  # Compute some operation

# Compute the gradient of `y` with respect to `w`
grad = g.gradient(y, w)

print(f"Gradient of y with respect to w: {grad}")
```

- Apply gradient on an image. $\alpha$ should be set in optimizer initialization already.

```python
optimizer.apply_gradients([(grad, generated_image)])
generated_image.assign(clip_0_1(generated_image))
```

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

- `tf.Tensor` is a tensor, an equivalent to numpy array with information for the computational graph.
- `tf.Variable(dtype)` it's best to specify the datatype here!

### Neural Net Operations

- Checking net layers: `vgg.layers`
- Checking layer output: `vgg.get_layer('block5_conv4').output`

## Data Loading

- Some pictures may have 4 channels: RGBA. To keep RGB:

```python
from PIL import Image
img_size = 400
content_image = np.array(
    Image.open("images/trout.png")
    .convert('RGB')
    .resize((img_size, img_size))
)
```

- In cases where we are loading the dataset from a directory,

```python
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
directory = "dataset/"
train_dataset = image_dataset_from_directory(directory,
                                            shuffle=True,
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE,
                                            validation_split=0.2,
                                            subset='training',
                                            seed=42)
validation_dataset = image_dataset_from_directory(directory,
                                            shuffle=True,
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE,
                                            validation_split=0.2,
                                            subset='validation',
                                            seed=42)

```

- In the case where we want to download a dataset:

```python
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
```
    - `prefetch()` partially downloads the dataset in a background thread, stores it in memory, and when `next(iter(train_dataset))` is called, it will download and prepare the next batch. This way, memory usage is reduced. `AUTOTUNE` finds an optimized buffer size.

- Data Augmentation

```python
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
from tensorflow.keras import Sequential
def data_augmenter():
    data_augmentation = Sequential()
    # RandomFlip, RandomRotation
    data_augmentation.add(RandomFlip('horizontal'))
    data_augmentation.add(RandomRotation(0.2)) #20% of a full circle
    return data_augmentation
```
    
- Visualize the dataset

```python
class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
```

- Load the data preprocess module for `mobilenet_v2`. This module does `x=x/127.5-1` so values are normalized to `[-1, 1]`

```python
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
```

## Model Definition

```python
def alpaca_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
    ''' Define a tf.keras model for binary classification out of the MobileNetV2 model
    Arguments:
        image_shape -- Image width and height
        data_augmentation -- data augmentation function
    Returns:
    Returns:
        tf.keras.model
    '''


    input_shape = image_shape + (3,)

    ### START CODE HERE

    base_model_path="imagenet_base_model/without_top_mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"

    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False, # <== Important!!!! so we are not using the output classifier
                                                   weights=base_model_path)


    # freeze the base model by making it non trainable
    base_model.trainable = None

    # create the input layer (Same as the imageNetv2 input size)
    # This is a symbolic placeholder that will get train_dataset when model.fit() is called
    inputs = tf.keras.Input(shape=input_shape)

    # apply data augmentation to the inputs
    x = data_augmentation(inputs)

    # data preprocessing using the same weights the model was trained on
    x = preprocess_input(x)

    # set training to False to avoid keeping track of statistics in the batch norm layer
    x = base_model(x, training=False)

    # add the new Binary classification layers
    # use global avg pooling to summarize the info in each channel
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # include dropout with probability of 0.2 to avoid overfitting
    x = tf.keras.layers.Dropout(0.2)(x)

    # use a prediction layer with one neuron (as a binary classifier only needs one)
    outputs = tf.keras.layers.Dense(1, activation=None)(x)

    ### END CODE HERE

    model = tf.keras.Model(inputs, outputs)

    return model
```

## Model Refining

The idea of refining a pretrained model is to have a small step size. Especially the later layers. This is because the early layers are earlier stage features, such as edges. Later features are more high level, like hair, ears, etc. So we want the high level features to adapt to the new data. So the way to do it is to unfreeze the model, set a layer to fine tune from, then freeze the model?? TODO

```python
base_model = model2.layers[4]   #?
base_model.trainable=True
print("number of layers: ", len(base_model.layers))
TRAINABLE_FROM=120
for l in base_model.layers[:TRAINABLE_FROM]:
    l.trainable=False

loss_function= tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.Adam(lr=0.1*base_learning_rate)
metrics=['accuracy']

model2.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)

fine_tune_epochs=5
total_epoch=initial_epoch+fine_tune_epochs
history_fine = model2.fit(
    train_dataset, epochs=total_epoch, initial_epoch=history.epoch[-1], validation_data=validation_dataset
)
```


## TODO

`tf.keras.losses.categorical_crossentropy(tf.transpose(labels), tf.transpose(logits), from_logits=True, )`

- `from_logits` is more numerically stable. We are trying to map the entire real number set using a floating number, that is, +- `2^32`. 

## Misc

- `HDF5` is "Hierarchical Data Format 5", a data format designed for compressing, chuking, and storing complex data hierarchies. It's similar to a filesystem, for example, you can create "groups" within an HDF5 file like creating a folder. Datasets are similar to files. HDF5 allows **access from multiple processes**, and is supported by multiple languages, like C, C++, Python.

