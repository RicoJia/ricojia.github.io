---
layout: post
title: Python - How To Write Pytest
date: '2019-02-01 13:19'
subtitle: Pytest Is The New Black (Compared To Unittest)
comments: true
header-img: "img/post-bg-2015.jpg"
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

### Using VSCode Debugger With Pytest

1. `ctrl+shift+p` choose `debug tests in the current file` or `debug all tests` (if you want to debug all tests under a configured directory)
2. In my debugger, I found that I have to manually set a breakpoint before the failure point in Pytest. (I might miss an easier route to get around this)
3. At the failed test, right click, and choose debug test

4. `pytest -s <my_test.py>` seems to be executing all test modules in the current directory: this will be enforced in a `pyproject.toml` environment.
