---
layout: post
title: Deep Learning - Hands-On ResNet Transfer Learning For CIFAR-10 Dataset
date: '2022-02-07 13:19'
subtitle: ResNet-50 Transfer Learning
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## GPU Set up

### Coursera Set Up

If you are a Coursera member, you can get free GPU access [in the Residual Networks assignment](https://www.coursera.org/learn/convolutional-neural-networks/home/week/3). As of Sept 25 2024, I can see:

```
%!nvidia-smi
Wed Sep 25 17:46:22 2024       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.05    Driver Version: 525.85.05    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A10G         On   | 00000000:00:1C.0 Off |                    0 |
|  0%   42C    P0   140W / 300W |  12940MiB / 23028MiB |     23%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

Here an `Nvidia A10G` is used. It's an industrial grade GPU: 150W, Ampere Architecture, It's 31.2 TF (FP32), and 250 TOPS (INT8).

#### SSL PITFALL

When I was trying to do

```python
from torchvision import models
model_ft = models.resnet18(weights='DEFAULT')
```

I saw:
```python
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth

---------------------------------------------------------------------------
SSLEOFError                               Traceback (most recent call last)
File /usr/lib/python3.8/urllib/request.py:1354, in AbstractHTTPHandler.do_open(self, http_class, req, **http_conn_args)
   1353 try:
-> 1354     h.request(req.get_method(), req.selector, req.data, headers,
   1355               encode_chunked=req.has_header('Transfer-encoding'))
   1356 except OSError as err: # timeout error
   ...
URLError: <urlopen error EOF occurred in violation of protocol (_ssl.c:1131)>
```

According to this post, it might be that Python version on the jupyter notebook does not support TLS1.1 +. I tried the solution there but to no avail. So for now, this notebook is **only good for Training from scratch, not for transfer learning.**

```python
import ssl
urlopen('https://www.howsmyssl.com/a/check', context=ssl._create_unverified_context()).read()
```

### Nvidia Orin Nano Setup

If you haven't please check out my blog on how to set up [Nvidia Orin Nano](../2024/2024-08-18-rgbd-slam-setup-nvidia-orin-nano.markdown). One pain point is Nano might run into insufficient memory issues. This can be lessened by choosing a smaller batch size.

## ResNet-50 Transfer Learning

### Data Loading

- ResNet was originally trained on the ImageNet dataset with 1.2M + high resolution images and 1000 categories. CIFAR-10 dataset has 60K 32x32 images across 10 classes. (Hence the 10) 
- `torch.backends.cudnn` is a CUDA Deep Neural Network Library has optimized primitives such as Convolution, pooling, activation funcs. MXNet, TensorFlow, PyTorch use this under the hood.
- `torch.backends.cudnn.benchmark` is A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
- `plt.ion()` puts pyplot into interactive mode: no explicit calls to plt.show(); show() is not blocking anymore, meaning we can see the real time updates.

```python
%matplotlib inline
# load the autoreload extension
%load_ext autoreload
# autoreload mode 2, which loads imported modules again 
# everytime they are changed before code execution.
# So we don't need to restart the kernel upon changes to modules
%autoreload 2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn #CUDA Deep Neural Network Library
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import time
import os
from tempfile import TemporaryDirectory

from PIL import Image
cudnn.benchmark = True
plt.ion()
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
```

For data transforms

- `scale=True` in ToDtype() scales RGB values to 255 or 1.0. 
- When downsizing, there could be **jagged edges** or **Moire Pattern** due to violation of the Nyquist Theorem. Antialiasing will apply low-pass filtering (smoothing out edges), resample with a different frequency
- `v2.Normalize(mean, std)` normalizes data around a mean and std, which does NOT clip under 1.0 (float) or 255 (int). This helps the training to converge faster, but visualization would require clipping in 1.0 or 255.

```python
DATA_DIR='./data'
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
transform_train = transforms.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5), #flip given a probability
    v2.ToImage(), # only needed if you have an PIL image
    v2.ToDtype(torch.float32, scale=True), #Normalize expects float input. scale the value?
    v2.Normalize(mean, std), #normalize with CIFAR mean and std
])
test_train = transforms.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True), #Normalize expects float input. scale the value?
    v2.Normalize(mean, std), #normalize with CIFAR mean and std
])
train_data = torchvision.datasets.CIFAR10(root=DATA_DIR, train = True, transform = transform_train, download = True)
test_data = torchvision.datasets.CIFAR10(root=DATA_DIR, train = False, transform = test_train, download = True)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)
```

For Dataloading:

- `torch.utils.data.Dataset` stores the data and their labels
- `torch.utils.data.DataLoader` stores an iterable to the data. You can specify batch size so you can create mini-batches off of it.
- some data are in CHW format (Channel-Height-Weight), so we need to flip it by `tensor.permute(1,2,0)`

```python
class_names = train_data.classes
def denorm(img):
    m = np.array(mean)
    s = np.array(std)
    img = img.numpy() * s + m
    return img

ROWS, CLM=2, 2
fig, axes = plt.subplots(nrows=ROWS, ncols=CLM)
fig.suptitle('Sample Images')
features, labels=next(iter(test_dataloader))
for i, ax in enumerate(axes.flat):
    img = denorm(features[i].permute(1,2,0).squeeze())
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(class_names[labels[i].item()])
plt.tight_layout()
plt.imshow(img)
```

## Model Definition

- Inplace operation uses no extra memory. So it's more friendly for large models

```python
self.relu=nn.ReLU(inplace=True)
```

- `nn.Sequential(*layers)` is a container that allows stacking of layers. The number of layers is determined during **runtime**. During forward pass, input x is fed through the layers in sequence. During backward pass, back prop is conducted in sequence as well. (this is the very definition of a "Sequential model")

- Batch norm layers `self.bn1` and `self.bn2` can't be shared because they have well, 4 different params each. (mean, variance, exponential decay's parameters )

- `nn.Identity()` is basically no-op.

- `m = nn.AdaptiveAvgPool2d((5, 7))`: given input `m x n x c`, output `5x7xc`.

- `x = torch.flatten(x, start_dim=1)` flattens `[batch_size, 64, 1, 1]` to `[batch_size, 64]`. 

### Model Training

- `model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))`: torch models actually could have tensors for GPUs. So if your model is trained on a GPU, it can't be loaded onto a CPU. This can be mitigated by `model.load_state_dict(torch.load(MODEL_PATH, map_location=device))`
- `loss.item()` gives the average loss across the current batch
- `model.eval()` and `model.train()`: to set dropout and batch normalization in the `eval` or `training` mode.