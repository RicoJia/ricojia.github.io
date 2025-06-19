---
layout: post
title: Python - Python Packaging System
date: '2019-02-06 13:19'
subtitle: Packaging, Pip
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

## Pip

### Externally Managed Environment

When you see “externally managed environment,” it means your system’s Python installation is controlled by **your OS’s package manager (e.g. Debian’s apt)**, not by pip. To keep the system stable, Debian (and other distros) disable pip’s ability to write into the global site-packages directory. That way, **all core Python libraries stay under apt’s control** and can’t be accidentally broken by a pip upgrade.

- “Core Python libraries” are the modules that ship as part of your Python installation
  - System Python `.deb` packages: `python3-stdlib`, `python3-distutils`, `python3-venv`, etc.

PEP 668 formalizes this: an “externally managed” Python cannot be modified by user-level pip installs.

- What is a "core python library"? Does this mean i can't use pip?

So, one can install using apt: `sudo apt install python3-ipykernel`
