---
layout: post
title: Python - Data Structures
date: 2019-02-09 13:19
subtitle: Dictionary, tuple
comments: true
header-img: img/post-bg-2015.jpg
tags:
  - Python
---
## Dictionary

- `setdefault` if value does exist, return the existing value; Otherwise, set the key with a new value.

```python
counts = {}

value = counts.setdefault("apples", 0)
print(value)   # 0
print(counts)  # {'apples': 0}
```

## Tuple

Python compares tuples lexicalgraphically, meaning, 12 > 10.

```
(12, 8.5, -0.23) > (10, 20.0, -0.01)
```
