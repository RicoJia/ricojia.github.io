---
layout: post
title: Python - How To Write Pytest
date: '2019-02-01 13:19'
subtitle: Pytest Is The New Black (Compared To Unittest)
comments: true
header-img: "img/home-bg-2015.jpg"
tags:
    - Python
---

## Assert

- For integer assertions:

```python
assert (1==1)
```

- For float assertions: 

```python
import pytest
1.0==pytest.approx(1.0)
```

- For numpy array assertions: 

```python
import numpy as np
array1 = np.array([1, 2, 3])
array2 = np.array([1, 2, 3])
np.testing.assert_allclose(array1, array2)
```