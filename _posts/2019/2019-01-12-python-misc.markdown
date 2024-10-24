---
layout: post
title: Python - Python Misc, os, shutil
date: '2019-01-03 13:19'
subtitle: Print
comments: true
header-img: "img/home-bg-2015.jpg"
tags:
    - Python
---

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
