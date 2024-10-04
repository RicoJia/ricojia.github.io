---
layout: post
title: Deep Learning - Image Segmentation
date: '2022-02-13 13:19'
subtitle: Encoder-Decoder, Fully-Convolutional Networks (FCN), U-Net
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Pre-Requitstes

### Encoder-Decoder framework

Autoencoders (or encoders) and autodecoders (or decoders) were introduced in the late 1980. An autoencoder compresses input data into smaller dimensions, and a decoder reconstructs them back into their original dimensions. Encoder-decoder is often useful for dimensionality reduction and feature learning.

Originally used for denoising and unsupervised tasks like dimensionality reduction.

### Fully Convolutional Networks, FCN (Long et al., UC Berkeley, CVPR 2014)

Fully Convolutional Networks (FCN) can output pixel-wise labels, this is also called **"Dense Prediction"**. The main innovation point is: They leverage a skip architecture to add **appearance information** from shallow, fine layers to a **semantic information** from deep, coarse layers. This mainly comes from the insight [1]:

> Semantic segmentation faces an inherent tension between semantics and location global information resolves what while local information resolves where.

To do pixel-perfect labelling:
1. Use shallow conv layers for feature extraction. In these layers, the height and width of feature maps go down (downsampling)
2. Output with dense layer cannot generate pixel-perfect labelling. Therefore, FCN upsamples from learned feature using [**transposed convolution.**](../2017/2017-01-07-transpose-convolution.markdown) The end of the next work is a Transposed Conv
3. Add skip connection betweeen the shallow conv layers and the later deep upsampling conv layers. The shallow layers have spatial information (where features are), but in conventional Conv-Dense hybrid architectures, that information is lost at the dense layers.
4. An **added bonus is the input size is not longer needed to be fixed** like architectures with dense layers.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/869ee426-9d5b-406e-be5e-77a4a8422a4b" height="300" alt=""/>
    </figure>
</p>
</div>

Choices of CNN can be VGG-16, AlexNet, etc. There are three types of outputs:

- FCN-32: the combined downsampling factor is 32. e.g., for an input `512x512`, the feature map that gets downsampled to is `16x16`. The feature map is then directly upsampled back to the original image size. This is very coarse, so FCN-16, FCN-8 are introduced
- FCN-16: The `16x16` feature map first gets upsampled to `32x32`, then it's added with a `32x32` coarse feature map with fine spatial information (from the Pool4 layer). Finally, it gets upsampled back to the original file size by a factor of 16.
- FCN 8: similar to FCN-16, but it finally gets upsampled by a factor of 8.

When combining shallow and deep layer outputs, FCN uses element-wise addition. It's simpler, memory over head is lower. 

Results

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/c8581a89-b4f5-4c39-b8e0-b14beb772fdb" height="200" alt=""/>
    </figure>
</p>
</div>

**Deficiencies of FCN:**

- Information loss at downsampling, so upsampling has trouble learning from it.

Fully convolutional computation is famously used in Semernet et al.'s OverFeat, a sliding window approach for object detection. A fully convolutional Network however is proposed by Long et al.

## U-Net (Ronneberger, U Freiburg, MICCAI, 2015)

U-Net is a pioneering image segmentation network primarily designed for tumor image segmentation.

The foundation of U-Net is the **encoder-decoder network**, and **FCN**[1]. The innovations in U-Net include:
1. Adding a skip connection between every matching downsampling and upsampling block. This allows U-Net to transfer low and high level information more comprehensively.  
2. Using a matching number of convolutions (for downsampling to feature maps) and transposed convolutions (for upsampling back to initial image size). This helps prevent model overfitting.
3. Shallow layers **learns local features** such as edges, corners. Their outputs are responses to the local features with high spatial fidelity (localization), but they lack a global understanding of the scene.
4. Deep layers, after multiple conv+pooling, have larger receptive fields. They capture global context (classes of objects that present in the image) better, but they need localization.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/1c3e2eab-5789-4ffb-a43f-86f0a440397d" height="300" alt=""/>
    </figure>
</p>
</div>

When combining shallow and deep layer outputs, U-Net **concatenates them together**. This preserves full information from both the encoder and the decoder.

### U-Net Architecture Summary

- Contracting Path (decoder):
    1. There are 2 3x3 conv+relu layers at each stage.
    2. Between each stage, there is downsampling by a factor of 2, max pooling with a stride of 2. In the meantime, the number of channels **doubles**
- Cropping Path: crops images from the contracting path, then concatenates to the feature map on the expanding path. TODO: why??? Loss of border pixel in convolution??
- Expansion Path (encoder):
    1. Upsamples feature maps by a factor of 2, using transpose convolution. Meanwhile decrease the number of channels.
    2. Concatenate cropped images from skip connections.
    3. Followed by 2 3x3 convolutions + ReLU

- Final Layer: 1x1 convolution to make the final output `nxnx11`, where 11 is the number of classes.

Finally, the output is `hxwxk`, where `k` is the output class number.

### U-Net Implementation TODO

```python

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate

from test_utils import summary, comparator

def conv_block(x, n_filters, kernel_initializer=he_initialization):
    l1 = conv_layer(n_filters=n_filters kernel_size=3, stride=1, padding='same', activation='relu')(x)
    l2 = conv_layer(n_filters=n_filters kernel_size=3, stride=1, padding='same', activation='relu')(x)
    # technically, there should be another conv layer with "same" padding.
    l3 = max_pool2D(kernel_size=2, stride=2)(2)
    # Don't forget Activation (not specified on paper)
    skip_output = l2
    return l3, skip_output


def upsampling_block(n_filters, kernel_size, kernel_initialization, x, skip):
    tf.keras.layers.Conv2DTranspose(
        n_filters,
        kernel_size,
        strides=(1, 1),
        padding='same'
        data_format=None,
        dilation_rate=(1, 1),
        activation='relu',
        use_bias=False,
    ) 

def upsampling_block(expansive_input, contractive_input, n_filters=32):
    pass

def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    pass

# Specify input
input = Input(input_size)

b1 = conv_block(Input, 64, dropout_prob=0., max_pooling=True)
b2 = conv_block(b1, 128, dropout_prob=0., max_pooling=True)
b3 = conv_block(b2, 256, dropout_prob=0., max_pooling=True)
b4 = conv_block(b3, 512, dropout_prob=0.3, max_pooling=True)
# bottleneck
b5 = conv_block(inputs=b4, n_filters=1024, dropout_prob=0.3, max_pooling=False) #? gets expanded into 1024, but then gets downsampled. 
b6 = upsampling_block(b5, contractive_input=b4, n_filters=512)
b7 = upsampling_block(b6, contractive_input=b3, n_filters=256)
b8 = upsampling_block(b7, contractive_input=b2, n_filters=128)
b9 = upsampling_block(b8, contractive_input=b1, n_filters=64)

conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(b9)
out = Conv2D(23, 1, padding=None)(conv9); # 1x1(b9, n_filters = 23). 1x1?
```

### Comments

- `he_initialization` -> `he_normal`
- Don't forget Activation (specified on paper), each conv layer should be followed by one.
- Skip connection is the output **before max pooling.**
- In for b4 and b5, add dropout_prob of 0.3
- `encoder, bottleneck, decoder`
- For the final conv, set dropout_prob to 0.3 again, and turn off max pooling

- **More epochs != better results** remember to experiment!

### Output And Training Data Labels

Image segmentation usually uses one-hot encoding as model output, but does NOT use one-hot encoding in training data's labels: one hot vectors are not memory efficient. Instead, we use uint8 as class labels for each pixel, **when each pixel belongs to ONE object**. There is a loss function that's designed for integer labels.

```python
unet.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```


Questions:
- Adding drop out? The paper mentions "Drop-out layers at the end of the contracting path perform further implicit data augmentation". In the meantime, data  the dataset for shift and rotation invariance.
tf.keras.layers.dropout
- Use cblock5 as expansive_input and cblock4 as contractive_input, with n_filters * 8. This is your bottleneck layer?
-  second element of the contractive block before the max pooling layer: isn't this the same as the input?
- why not use one-hot encoding in image seg? you just use an integer?
- What is sparse categorical loss?

## Op Determinism In TensorFlow TODO

[Op Determinisim](https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/config/experimental/enable_op_determinism)

Training Code to get consistent result:

```python
EPOCHS = 15
VAL_SUBSPLITS = 5
BUFFER_SIZE = 500
BATCH_SIZE = 32

tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()
train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(processed_image_ds.element_spec)

unet = unet_model((img_height, img_width, num_channels))
unet.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model_history = unet.fit(train_dataset, epochs=EPOCHS)
```



## References

[1] Long, J., Shelhamer, E., and Darrell, T. 2015. Fully Convolutional Networks for Semantic Segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 3431–3440.

[2] Ronneberger, O., Fischer, P., and Brox, T. 2015. U-Net: Convolutional Networks for Biomedical Image Segmentation. In Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI). 234–241.
