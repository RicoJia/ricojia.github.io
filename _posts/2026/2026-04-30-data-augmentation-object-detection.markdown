---
layout: post
title: Data Augmentation For Object Detection
date: 2026-04-30 13:19
subtitle: ""
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine-Learning
---

## Training

1. Label a small but diverse calipers dataset.  
2. Fine-tune RF-DETR Large.  
3. Run it on new/unlabeled videos.
4. Find where it fails.
5. Add those failed frames back into the training set.
6. Retrain

- False negative: Calipers are visible, but RF-DETR does not detect them. Add this frame to training with a correct calipers box.
- False positive No calipers are present, but the model says something is calipers. Add this frame as a **negative frame**: image with zero boxes.

Tools: (5000 images)

- Blender / BlenderProc: Very common for research.
- NVIDIA Isaac Sim / Omniverse
- a simpler OpenGL / pyrender style pipeline can work.
  - Render the CAD over random backgrounds. object render + real background image compositing

Then you have: `http://real_train/real_val/real_test/`

## Data Augmentation

- Randomize:

```
matte grayblack plasticmetallic grayslightly rough surfaceslightly glossy surfacedifferent brightnessdifferent background color
```

- Fix:
  - camera distance
    - 50-60%: full object visible, centered-ish
    - 20-30%: object smaller, with table/background10-20%: partial crop / occlusion / close-up object occupies 20-70% of image width

  - object scale in image
  - full vs cropped views
  - background/tablebbox correctness

get segmentation:

```python
writer.initialize(
    output_dir=OUTPUT_DIR,
    rgb=True,
    bounding_box_2d_tight=True,
    semantic_segmentation=True,
)
```
