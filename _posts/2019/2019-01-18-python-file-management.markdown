---
layout: post
title: Python - File Management
date: '2019-01-18 13:19'
subtitle: os
comments: true
header-img: "img/post-bg-2015.jpg"
tags:
    - Python
---

## File Path Processing

- `os.path.basename("/home/foo/Downloads/000000.pcd")` gives the string after the last "/". So here we get: `000000.pcd`
- `os.path.splitext("/home/foo/Downloads/000000.pcd")` gives `'/home/foo/Downloads/000000', '.pcd'`

The Path library

```python
p = Path('/home/halo_runtime/install/velodyne_msgs/share/velodyne_msgs/ros1_msg/VelodyneScan.msg')

p.name      # 'VelodyneScan.msg'  (base filename)
p.stem      # 'VelodyneScan'      (without extension)
p.suffix    # '.msg'              (extension)
p.suffixes  # ['.msg']            (all extensions, useful for e.g. '.tar.gz')
```
