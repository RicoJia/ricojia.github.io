---
layout: post
title: Python - Functools
date: 2019-01-27 13:19
subtitle: lru_cache
comments: true
header-img: img/post-bg-2015.jpg
tags:
  - Python
---
---

## lru_cache

One subtle but very real bug when using `functools.lru_cache` with mutable objects is **cache poisoning** caused by mutability. The cache stores and returns the exact same object references on every call. If a cached function returns the object and a caller later modifies it in place, the cached value itself becomes corrupted. For example:

```python
@lru_cache(max_depth=1)
def icosahedron_2_sphere(level:int):
 return Obj

V, F = icosahedron_2_sphere(2)  
V /= 2  # in-place modification — corrupts the cached array
```

 `V /= 2` mutates the array in place,  whereas `V = V/2` creates a new array.
