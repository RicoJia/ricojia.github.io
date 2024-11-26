---
layout: post
title: Robotics - ROS1 Message Processing
date: '2024-01-03 13:19'
subtitle: Image Data
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS1
comments: true
---

## Data Manipulation

### Image Data

- Read, manipulate, and output image data

```python
from sensor_msgs.msg import Image
import numpy as np
def get_image(msg):
    data = msg.data
    height = msg.height
    width = msg.width
    encoding = msg.encoding
    if encoding == "rgb8":
        image = np.frombuffer(data, np.uint8).reshape((height, width, 3))
    elif encoding == "bgr8":
        image = np.frombuffer(data, np.uint8).reshape((height, width, 3))[:, :, ::-1]
    elif encoding == "mono8":
        image = np.frombuffer(data, np.uint8).reshape((height, width))
    else:
        raise NotImplementedError(f"Encoding {encoding} not supported.")
    do_stuff()
    if encoding in ["rgb8", "bgr8"]:
        # Ensure masked_output has shape (H, W, 3)
        if masked_output.ndim == 2:
            # If mask is single channel, stack to make it 3-channel
            masked_output = np.stack([masked_output] * 3, axis=-1)
        elif masked_output.shape[2] == 1:
            masked_output = np.concatenate([masked_output] * 3, axis=2)
    elif encoding == "mono8":
        # Ensure masked_output has shape (H, W)
        if masked_output.ndim == 3:
            masked_output = masked_output.squeeze(2)

    msg.data = masked_output.flatten()  # DO NOT USE tobyte(),
```
