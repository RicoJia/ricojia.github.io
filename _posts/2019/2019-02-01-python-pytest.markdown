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

## Run Pytest

- Run a specific test file `pytest path/to/test_file.py`
- Run a test in it: `pytest path/to/test_file.py::test_function_name`

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

## Test Fixture

The boiler plate is:

```python
import pytest
@pytest.fixture
def basic_config():
    return {
        "batch_size": 2,
    }
def test_og_positional_encoder(basic_config):
    batch_size = basic_config["batch_size"]
    ...
```

## Patching

### Patching a Function

To patch my_func() from MyModule, pytest requires it to be a `MagicMock` object. This means we must specify return_value and pass mock_my_func into the test function.

```python
from unittest.mock import patch
import requests

@patch('MyModule.my_func', return_value=my_func_patch())
def test_get_license_hash(mock_my_func):
    """Test if the server responds with 200 OK."""
    response = requests.get(f"{SERVER_URL}/GetLicenseHash")
    assert response.status_code == 200
```

### Patching a Constant or List

Unlike functions, patching constants or lists does not require a `MagicMock`. Instead, patch directly replaces the original object with a real one, meaning no return_value or mock object argument is needed.

```python
@patch('MyModule.WEBCAM_RESTART_COMMAND', ["echo", "hello"])
def test_webcam_restart():
    """Test if the webcam restart command is patched correctly."""
    response = requests.get(f"{SERVER_URL}/RestartService")
    assert response.status_code == 200
```