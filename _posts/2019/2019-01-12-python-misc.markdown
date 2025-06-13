---
layout: post
title: Python - Python Misc, os, shutil
date: '2019-01-03 13:19'
subtitle: Sys, Print, Argparse Path-Related Utils, HDF5
comments: true
header-img: "img/post-bg-2015.jpg"
tags:
    - Python
---

## `sys`

- Check current python version

```python
if sys.version_info.major == 3 and sys.version_info.minor == 10:
    print("Python version is 3.10")
else:
    print(f"Python version is not 3.10, it's {sys.version_info.major}.{sys.version_info.minor}")
```

## Numbers

- Not a number `float('nan')`. This is often seen in `0/0`
  - In numpy, this is `np.nan`. To check, it's `numpy.isnan()`

## Print Options

- Multiline Printing

```python
print(f'''
line 1
line2
''')
```

- Print float with 2 decimal digits

```python
print(f"{test_acc:.2f}%")
```

## Argparse

- Adding a required arg:

```python
parser.add_argument("--name", "-n", type=str, required=True, help="Name of a bag in rgbd_slam_rico/data")
```

## Path Related

### `os` Library

- Absolute Path

```python
script_path = os.path.abspath(__file__)
```

- Remove a file if it exists:

```python
if os.path.exists(output_path):
    os.remove(output_path)
```

### `shutil`

- remove non-empty directory:

```python
shutil.rmtree(DIR)
```

## For-Else Loop

```python
k = 10
for i in range(k):
    print(k)
    break
else:
    print("hello")
```

### HDF5

- open-source file for **store large amount of data**
    - largely used in numpy community
    - Can handle many different datatypes in multiple data blocks, with metadata
    - We can access them in data blocks
    - numpy array holds the data, and directly to and from the data.
    - used as ML model weights
- To use HDF5, numpy needs to be imported.

