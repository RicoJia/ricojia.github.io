---
layout: post
title: Deep Learning - PyTorch Data Loading
date: '2022-02-21 13:19'
subtitle: RESNET-50 Data Loading, Data Transforms, Custom Data Loading
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## RESNET-20 Example

### Imports

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

- `torch.backends.cudnn` is a CUDA Deep Neural Network Library has optimized primitives such as Convolution, pooling, activation funcs. MXNet, TensorFlow, PyTorch use this under the hood.
- `torch.backends.cudnn.benchmark` is A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
- `plt.ion()` puts pyplot into interactive mode: no explicit calls to plt.show(); show() is not blocking anymore, meaning we can see the real time updates.

- ResNet was originally trained on the ImageNet dataset with 1.2M + high resolution images and 1000 categories. CIFAR-10 dataset has 60K 32x32 images across 10 classes. (Hence the 10)

### Data Loading

```python
DATA_DIR='./data'
mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]
transform_train = transforms.Compose([
    # v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomCrop(32, padding=4),
    v2.RandomHorizontalFlip(p=0.5), #flip given a probability
    v2.ToImage(), # only needed if you don't have an PIL image
    v2.ToDtype(torch.float32, scale=True), #Normalize expects float input. scale the value?
    v2.Normalize(mean, std), #normalize with CIFAR mean and std
])
transform_test = transforms.Compose([
    # v2.RandomResizedCrop(size=(224, 224), antialias=True),
    # v2.RandomHorizontalFlip(p=0.5), #flip given a probability
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True), #Normalize expects float input. scale the value?
    v2.Normalize(mean, std), #normalize with CIFAR mean and std
])
train_data = torchvision.datasets.CIFAR10(root=DATA_DIR, train = True, transform = transform_train, download = True)
test_data = torchvision.datasets.CIFAR10(root=DATA_DIR, train = False, transform = transform_test, download = True)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=1, pin_memory=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)
```

For data transforms

- `scale=True` in `ToDtype()` scales RGB values to 255 or 1.0.
- When downsizing, there could be **jagged edges** or **Moire Pattern** due to violation of the Nyquist Theorem. Antialiasing will apply low-pass filtering (smoothing out edges), resample with a different frequency
- `v2.Normalize(mean, std)` normalizes data around a mean and std, which does NOT clip under 1.0 (float) or 255 (int). This helps the training to converge faster, but visualization would require clipping in 1.0 or 255.
- `torch.utils.data.Dataset` stores the data and their labels
- `torch.utils.data.DataLoader` stores an iterable to the data. You can specify batch size so you can create mini-batches off of it. By default, it returns data in `CHW` format
- some data are in CHW format (Channel-Height-Weight), so we need to flip it by `tensor.permute(1,2,0)`

We normalize the pixel values to [0,1], then subtract out the [mean and std of the CIFAR-10 dataset](https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data). It's a common practice to normalize input data so images have consistent data distributions over RGB channels (imaging very high and low values)

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

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/9f2db0cd-ba6f-4162-a58b-d5ba87483496" height="200" alt=""/>
        <figcaption>Normalized Input Data</figcaption>
    </figure>
</p>
</div>

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/a3f33658-1795-4a9a-8e99-7cb5874e3a41" height="200" alt=""/>
        <figcaption>Regular Input Data</figcaption>
    </figure>
</p>
</div>

## Custom Data Loading

For image classification, one custom way to store images is to save images under directories named with its class. Then, save a label -> class name mapping.

Here, we are loading PASCAL VOC (Visual Object Classification) 2007 Dataset to test a neural net trained for CIFAR_10. Some key nuances include:

- CIFAR-10 takes in `32x32` images and we need to supply some class name mappings. Input data normalization is done as usual
- We do not add images and their labels if the labels don't appear in the class name mapping
- A custom `torch.utils.data.Dataset` needs to subclass `Dataset` and has  `__len__(self)` and `__getitem__(self)` methods.

```python
import torch
from torch.utils.data import Dataset
voc_root = './data'
year = '2007'
transform_voc = transforms.Compose([
    v2.Resize((32,32)),
    v2.ToTensor(),
    v2.Normalize(mean, std), #normalize with CIFAR mean and std
])

# These are handpicked VOC->CIFAR-10 mapping. If VOC's label doesn't fall into this dictionary, we shouldn't feed it to the model.
class_additional_mapping = {'aeroplane': 'airplane', 'car': 'automobile', 'bird':'bird', 'cat':'cat', 'dog':'dog', 'frog':'frog'}

mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]

class FilteredVOCtoCIFARDataset(Dataset):
    def __init__(self, root, year, image_set, transform=None, class_mapping=None):
        self.voc_dataset = torchvision.datasets.VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=True,
            transform=None  # Transform applied manually later
        )
        self.transform = transform
        self.class_mapping = class_mapping
        self.filtered_indices = self._filter_indices()

    def _filter_indices(self):
        indices = []
        for idx in range(len(self.voc_dataset)):
            target = self.voc_dataset[idx][1]  # Get the annotation
            objects = target['annotation'].get('object', [])
            if not isinstance(objects, list):
                objects = [objects]  # Ensure it's a list of objects
            if len(objects) > 1:
                continue
            obj = objects[0]
            label = obj['name']
            if label in self.class_mapping:  # Check if class is in our mapping
                indices.append(idx)
        return indices

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        actual_idx = self.filtered_indices[idx]
        image, target = self.voc_dataset[actual_idx]
        
        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        # Map VOC labels to CIFAR-10 labels
        objects = target['annotation'].get('object', [])
        if not isinstance(objects, list):
            objects = [objects]  # Ensure it's a list of objects

        # Create a list of labels for the image
        labels = []
        for obj in objects:
            label = obj['name']
            if label in self.class_mapping:
                labels.append(self.class_mapping[label])

        # Return the image and the first label (as a classification task)
        return image, labels[0]  # In classification, return a single label per image

dataset = FilteredVOCtoCIFARDataset(
    root=voc_root,
    year='2007',
    image_set='val',
    transform=transform_voc,
    class_mapping=class_additional_mapping
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,          # Adjust based on your memory constraints
    shuffle=True,
    num_workers=2,         # Adjust based on your system
    pin_memory=True
)
```