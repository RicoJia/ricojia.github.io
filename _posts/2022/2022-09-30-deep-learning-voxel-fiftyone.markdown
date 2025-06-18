---
layout: post
title: Deep Learning - Voxel Fiftyone
date: '2022-06-18 13:19'
subtitle: TODO
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Fifty-one

### Features Of `fiftyone`

- `pip install fiftyone`

```python
import fiftyone as fo
dataset = fo.Dataset("my_dataset")
sample = fo.Sample(filepath="/path/to/image.png")
sample.tags.append("train")
sample["custom_field"] = 51

dataset.add_sample(sample)

view = dataset.match_tags("train").sort_by("custom_field").limit(10)

for sample in view:
    print(sample)
```

- 🗃️ Dataset: Think of it as a container for your data, composed of individual Sample objects.
  - 📌 Sample: Represents wøa single data point (e.g., an image). Stores essential info like filepaths and custom fields.
  - Each sample has an ID, file path, tags, label, field
  - 🧰 Custom Fields: Add flexibility! Store anything from basic types (numbers, strings) to complex labels (bounding boxes, segmentations).
- 🪄 Automatic Media Type Detection: FiftyOne infers if a sample is an image, video, etc., based on its filepath.

- The Dataset is backed by a MongoDB database

- are there 3D point cloud features?
  - Clip model: reduce what?
  - What is data curation:
    - Search, filter by text.
    - find similar images

Correspondent: Antonio Rueda Toicen
