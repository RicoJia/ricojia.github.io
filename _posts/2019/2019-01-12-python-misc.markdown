---
layout: post
title: Python - Python Misc, os, shutil
date: '2019-01-03 13:19'
subtitle: Print
comments: true
header-img: "img/post-bg-2015.jpg"
tags:
    - Python
---

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

## shutil

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
