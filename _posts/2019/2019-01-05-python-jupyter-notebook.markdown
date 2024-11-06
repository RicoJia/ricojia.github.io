---
layout: post
title: Python - Jupyter Notebook Common Magic Commands
date: '2019-01-05 13:19'
subtitle: Jupyter Notebook Common Magic Commands
comments: true
header-img: "img/post-bg-2015.jpg"
tags:
    - Python
---

Magic command `%command` is a **function in IPython**.

- Make sure `matplotlib` is rendered inline
- Immediately autoreload modules if they are changed outside

```python
%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load the autoreload extension
%load_ext autoreload
# autoreload mode 2, which loads imported modules again 
# everytime they are changed before code execution.
# So we don't need to restart the kernel upon changes to modules
%autoreload 2
```

- **%timeit**: runs the command under test multiple loops, each loop multiple times to get the Gaussian statistics of its speed. 

```python
from logging import getLogger
logger = getLogger("Logger")
# see 574 μs ± 55.2 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
%timeit logger.warn("hello_world")
```
