---
layout: post
title: Computer Vision - Non Maximum Suppression
date: '2021-01-05 13:19'
subtitle: Non Maximum Suppression (NMS)
header-img: "home/bg-o.jpg"
comments: true
tags:
    - Computer Vision
---

In image classification, we often get many bounding boxes of the same object in a small subarea. Non Maximum Suppression (NMS) can help us pick which boxes we want to keep.

Inputs:

- Bounding box coordinates. These bounding boxes may / may not be far from each other. 
- Confidence score of each bounding box.
- A threshold for determining if two bounding boxes are "separate enough", IOU (Intersection Over Union)

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/662bcbcc-50d3-43fc-86e0-97fb01c7d5cb" height="300" alt=""/>
        <figcaption><a href="https://www.linkedin.com/pulse/non-max-suppression-object-detection-nabeelah-maryam-2gr7f/">Source: Nabeelah Maryam</a></figcaption>
    </figure>
</p>
</div>

The basic idea is to go through every pair of bounding boxes, following the descending order of confidence scores, and filter out the bounding boxes that has high intersection over union (IoU) ratios. The remaining bounding boxes are the ones that are separate from each other and have high confidence scores. 

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/212cfcc3-b6cb-49bd-9197-9b1c850147e1" height="300" alt=""/>
        <figcaption><a href="https://www.google.com/url?sa=i&url=https%3A%2F%2Fkorlakuntasaikamal10.medium.com%2Fintersection-over-union-a8e04c3d03b3&psig=AOvVaw1Goh-uTS9ihTL4MTDw-fX2&ust=1723930055452000&source=images&cd=vfe&opi=89978449&ved=0CBcQjhxqFwoTCIijnYm6-ocDFQAAAAAdAAAAABAE">Source: Kamal_DS </a></figcaption>
    </figure>
</p>
</div>

Well, let's jump right into the code!
- First, delete the boxes with a confidence lower than 0.6

```python
#! /usr/bin/env python3

import numpy as np
import typing

def handcrafted_non_maximum_suppresion(scores: np.ndarray, boxes: np.ndarray, overlapping_thre: float = 0.3) -> typing.List:
    # Sort scores into descending order and store their indices
    score_order = scores.argsort()[::-1]
    
    # Store individual coordinates in lists.
    x1s = boxes[:, 0]
    y1s = boxes[:, 1]
    x2s = boxes[:, 2]
    y2s = boxes[:, 3]

    # Find areas of each box
    areas = (x2s - x1s + 1) * (y2s - y1s + 1)
    keep = []

    while score_order.size > 0:
        keep.append(score_order[0])
        # Find coords differences along x
        other_x2_this_x1 = x2s[score_order[1:]] - x1s[score_order[0]]
        this_x2_other_x1 = x2s[score_order[0]] - x1s[score_order[1:]]
        # Rest of the score_order elements
        x_overlapping_lengths = np.maximum(np.minimum(other_x2_this_x1, this_x2_other_x1), 0)

        other_y2_this_y1 = y2s[score_order[1:]] - y1s[score_order[0]]
        this_y2_other_y1 = y2s[score_order[0]] - y1s[score_order[1:]]
        y_overlapping_lengths = np.maximum(np.minimum(other_y2_this_y1, this_y2_other_y1), 0)

        # Find intersection area
        overlapping_areas = x_overlapping_lengths * y_overlapping_lengths
        ious = overlapping_areas / (areas[score_order[0]] + areas[score_order[1:]] - overlapping_areas)

        independent_box_indices = np.where(ious <= overlapping_thre)[0]
        # Because ious, independent_box_indices are arrays starting at the next element, we want to add 1 here.
        score_order = score_order[independent_box_indices+1]

    return keep

# [x1, y1, x2, y2]
boxes = np.array([
    [100, 100, 210, 210],
    [101, 101, 209, 209],
    [105, 105, 215, 215],
    [150, 150, 270, 270]
])

scores = np.array([0.8, 0.79, 0.75, 0.9])
overlapping_thre = 0.3
selected_indices = handcrafted_non_maximum_suppresion(scores=scores, boxes=boxes, overlapping_thre=overlapping_thre)
# Should see [3,0]
print(selected_indices)
```

- IoU threshold is often 0.5