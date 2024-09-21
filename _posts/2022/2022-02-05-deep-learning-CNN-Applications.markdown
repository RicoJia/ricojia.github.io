---
layout: post
title: Deep Learning - CNN Applications
date: '2022-02-03 13:19'
subtitle: TensorFlow Keras Sequential and Functional Models
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## TF Keras Sequential API

TF Keras Sequential API can be used to build simple feedforward, sequential models.

```python
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

model = tf.keras.Sequential([
        ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
        ## Need to specify input shape 
        tf.keras.layers.ZeroPadding2D(
            padding=(3, 3), 
            input_shape=(64,64, 3)
        ),
        ## Conv2D with 32 7x7 filters and stride of 1
        tf.keras.layers.Conv2D(
                filters = 32,
                kernel_size = 7,
                strides = 1,
                data_format="channels_last",
        ),
        ## BatchNormalization for axis 3 (-1 is RGB. We want them separate of course)
        tf.keras.layers.BatchNormalization(
            axis=3
        ),
        ## ReLU. There's a function  tf.keras.activations.relu. Don't use that one
        tf.keras.layers.ReLU(),
        ## Max Pooling 2D with default parameters
        tf.keras.layers.MaxPool2D(),
        ## Flatten layer
        tf.keras.layers.Flatten(),
        ## Dense layer with 1 unit for output & 'sigmoid' activation
        tf.keras.layers.Dense(
            units=1,
            input_shape=(32768,),
            activation='sigmoid',
        )
    ])
    
happy_model = happyModel()
# Print a summary for each layer
for layer in summary(happy_model):
    print(layer)

    
happy_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

happy_model.summary()
happy_model.fit(x_train, y_train, epochs=10, batch_size=16)

test_loss, test_acc = happy_model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

predictions = model.predict(x_new)
```

- `compile()` is to set up the optimizer, loss function, and finally the computational graph. It's analogous to "compilation" so the model is ready to be trained.

### Limitations of the Sequential API

1. There's no layer sharing.
2. Each layer has a single input and a single output.

So more complex architectures such as residual connections or multiple branches cannot be modelled by the Sequential API.

## The TF Keras Functional API

The Functional API is more flexible than Sequential. It is able to build models with non-linear topologies (e.g., the skip connection), shared layers, as well as layers with multiple inputs and outputs. **The Sequential model is a straight line, whereas the Functional is a graph**.

```python
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)

input_shape = (64, 64, 3)
input_img = tf.keras.Input(shape=input_shape)
## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
Z1 = tf.keras.layers.Conv2D(filters= 8 , kernel_size= 4 , padding='same')(input_img)
## RELU
A1 = tf.keras.layers.ReLU()(Z1)
## MAXPOOL: window 8x8, stride 8, padding 'SAME'
P1 = tf.keras.layers.MaxPool2D(pool_size=(8,8), strides=8, padding='same')(A1)
## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
Z2 = tf.keras.layers.Conv2D(filters= 16 , kernel_size= 2 , padding='same')(P1)
## RELU
A2 = tf.keras.layers.ReLU()(Z2)
## MAXPOOL: window 4x4, stride 4, padding 'SAME'
P2 = tf.keras.layers.MaxPool2D(pool_size=(4,4), strides=4, padding='same')(A2)
## FLATTEN
F = tf.keras.layers.Flatten()(P2)
## Dense layer
## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
outputs = tf.keras.layers.Dense(units=6, activation='softmax')(F)


model = tf.keras.Model(inputs=input_img, outputs=outputs)

model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
model.summary()
history = model.fit(train_dataset, epochs=100, validation_data=test_dataset)

# This could take a short while, visualizing loss over time
history.history

# Model Accuracy
df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
```

## Model Subclassing

While the Functional API gives complex but well defined models, model subclassing is a technique that allows even more flexible modelling (e.g., RNNs, custom training loops, dynamic models, etc.)
