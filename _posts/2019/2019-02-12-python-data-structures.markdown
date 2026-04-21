---
layout: post
title: Python - Data Structures
date: 2019-02-09 13:19
subtitle: Dict
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
