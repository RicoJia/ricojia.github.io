---
layout: post
title: Python - Python Packaging System
date: '2019-02-06 13:19'
subtitle: Packaging
comments: true
header-img: "img/post-bg-2015.jpg"
tags:
    - Python
---

## Setuptools

### Entrypoints

Entrypoints are a way to specify components in your package to the rest of the python space, so they can be dynamically discovered. So this enables:

- Console scripts (command line tools): executable commands. `setuptools` generates binaries in your Python environment's `bin`, so they can be run.
- Plugins
- Custom Extensions

Example `setup.py`:

```python
from setuptools import setup, find_packages
setup(
    name='your_package_name',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'your_command = your_package.module:function',
        ],
        # Other entry point groups can be defined here
    },
)
```

- In ROS2, the most commonly used entrypoints are **console scripts**.
