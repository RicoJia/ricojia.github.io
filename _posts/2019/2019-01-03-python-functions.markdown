---
layout: post
title: Python - Functions
date: '2019-01-03 13:19'
subtitle: Map-Reduce-Filter, Lambda (Under Active Updates)
comments: true
header-img: "img/home-bg-2015.jpg"
tags:
    - Python
---

## Map, Reduce, Filter

Map, Reduce, Filter are paradigms of functional programming. WIth them, we can write simpler and shorter programs. They are applied on iterables. 

### Map

`map(callable, iterable)` applies callable to every iterable.

Example: Add 1 to each element in the list.

```python
# Unpack map obj (applies the function on everything in the list)
x,y,z = map(lambda x: x+1, [1,2,3]) 
```

### Reduce 

`reduce(callable, iterable)` can apply callable on `iterable`, **accumulatively**, meaning in the below example, the lambda is applied on each item and the current accumulated result.

```python
from functools import reduce
ls = [1,2,3,4,5]
# see 18, the sum of ls
print(reduce(lambda x, y: x+y, ls))
```

### Filter

Filter, as its name suggests, returns a `filter(predicate, iterable)` objects that can be casted into a "leaned down" iteration, after applying `predicate`.

```python
di = {1:"1", 2:"2", 3:"3"}
# converts to [(1, "1"), (2, "2"), (3, "3")]
di_list = list(di.items())
print(list(filter(lambda x: x[0] < 3, di_list)))
```

## Lambda Function

A lambda expression is a function without name.

```python
#input is x, return x+1
b = lambda x:x+1
print b(1)
```

## Functools

### partial

-  `functools.partial` returns a wrapper with some args bound to given values.

```python
# 1 partial - we just need to bind the function with any keyworded args
from functools import partial
def func(a, b): 
    print("func: ", a, b)
func_w = partial(func, b = 12)
func_w(a = 13)
```

- Its equivalent implementation is

```python
def rico_partial(func, *args, **kwargs):
    # simplified version
    # def wrapper(a): 
    #     # kwargs here is a dict, need to unpack it
    #     return func(a, **kwargs)
    # return wrapper
    def wrapper(*extra_args, **extra_kwargs):
        # need nonlocal since we are reusing args, and kwargs, which will be true local vars
        nonlocal args, kwargs
        # args here is a tuple already
        args = list(args)
        args.extend(extra_args)
        kwargs = {**kwargs, **extra_kwargs}
        return func(*args, **kwargs)
    return wrapper
rico_func_w = rico_partial(func, b = 12)
rico_func_w(a=13)
```