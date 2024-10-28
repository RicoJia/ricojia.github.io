---
layout: post
title: Deep Learning - Data Augmentations
date: '2022-05-14 13:19'
subtitle: Albumentations
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Pre-processing

### Shuffle Data

```python
shuffled_main_dataset = torch.utils.data.Subset(
    main_dataset,
    torch.randperm(dataset_size)
)

# train_dataset is a Subset object
# main_dataset becomes train_dataset.dataset

class_num = len(shuffled_main_dataset.dataset.classes)
```

## Albumentations

`Albumentations` is a library for pixel-wise image augmentations. It was developed by a few Kaggle experts, masters, and grandmasters. 

For the full list of augmentations, please see [here](https://github.com/albumentations-team/albumentations?tab=readme-ov-file#list-of-augmentations)

pip install albumentations

```
import albumentations as A

# Define the augmentation pipeline, and add mask as an input arg during object initiation
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ElasticTransform(p=0.2),
    # Add more augmentations as needed
], additional_targets={'mask': 'mask'})


augmented = transform(image=image, mask=mask)
augmented_image = augmented['image']
augmented_mask = augmented['mask']
```

- Standard practice in PyTorch is to augment in `Dataset.__getitem()__`.
