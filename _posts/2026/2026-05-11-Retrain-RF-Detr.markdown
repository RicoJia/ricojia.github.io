---
layout: post
title: "[Robotics] Retrain RF DETR"
date: 2026-05-11 13:19
subtitle:
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine-Learning
---
## RF-DETR Fine-Tuning for a Custom Class

When fine-tuning RF-DETR on a custom object, the usual workflow is:

```text
pretrained RF-DETR backbone/features
-> replace or reinitialize the classification head
-> train on the custom dataset
```

For example, if the target object is `motor_sleeve`, the new classifier head is trained for the custom label space:

```text
0 motor_sleeve
background / no-object
```

It is **not** using the old COCO label space:

```text
0 person
1 bicycle
...
67 cell phone
...
100 motor_sleeve
```

The old meaning of `class 0 = person` existed only in the original COCO classification head. During one-class fine-tuning, that head is resized or replaced, so `class 0` now means `motor_sleeve` in the new checkpoint.

There is no numeric class-ID collision inside the fine-tuned checkpoint.

The pretrained model still keeps useful visual features in the backbone, such as edges, textures, shapes, and object-level representations. However, the final classification layer is retrained for the new dataset.

The main risk is not class-ID collision. The real risk is **false-positive confusion**. For example, the model may incorrectly detect visually similar objects as `motor_sleeve` if the custom dataset does not include enough negative examples or background variation.

If the goal is to detect both COCO classes and the new custom object, then the training setup must be different:

```text
0 person
1 bicycle
...
67 cell phone
...
N motor_sleeve
```

In that case, the model needs a multi-class head and training data for both the original classes and the new class. Otherwise, the model may forget the old classes.

For the current one-class fine-tuning setup, the model should be treated as:

```text
motor_sleeve detector + background/no-object
```

not as a COCO detector with one extra class.
