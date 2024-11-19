---
layout: post
title: Deep Learning - MobilenetV2 Hands-On
date: '2022-08-01 13:19'
subtitle: Inverted Skip Connection
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
    - Hands-On
---

## Data

We are using the COCO dataset to train a multi-class classifier with a MobileNetV2. Therefore:

- We need a multi-hot vector where a class is 1.0 if it's in the class list

## Model

- At the end of the model, we have one dense layer as a "classifier".
  - The classifier does NOT have sigmoid at the end. This is trained for multi-label classification. There should be a sigmoid (not softmax) layer following it, but it is handled in the nn.BCEWithLogitsLoss.
  - The classifier dim is the same as number of classes in the dataset. So if we train across multiple datasets, we have to map the datasets correctly.
- Num Parameters?

## Training

- Mixed precesion

## Final Results

With a probability decision threshold being `0.5`:

- the raw training set Accuracy is `97.97%`. The validation set accuracy is `97.73%`
- Recall values are: training set 53.47789%, validation set:  48.65012%
