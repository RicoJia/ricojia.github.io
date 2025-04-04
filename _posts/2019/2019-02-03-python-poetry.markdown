---
layout: post
title: Python - Poetry Packaging System and Pip
date: '2019-02-03 13:19'
subtitle: Poetry, Pip
comments: true
header-img: "img/post-bg-2015.jpg"
tags:
    - Python
---

## Introduction

Using Poetry is highly recommended because it simplifies dependency management and packaging, even when not using virtual environments. Environments without Poetry can install poetry packages because source distributions `sdist` and wheel files `.whl` are standard Python packages formats. Also, by using poetry, we can avoid manually tweaking several files required by `setuptools`; `setup.py`, `setup.cfg`, `MANIFEST.in`

I'm using Docker and generally don't use `venv`.

## Sample Workflow

I can disable venv by:

```python
poetry config virtualenvs.create false
```

Then, initialize the environment. You will be prompted to type in dependencies

```python
poetry init
```

Later, to add dependencies:

```python
poetry add requests numpy
```

A `poetry.lock` ensures consistent dependencies across builnds. Now if your environment has poetry, do `poetry install`

To add a dependency: `poetry add`

## Push To PyPi

- File Structure

```python
your_package/
├── your_package/
│   ├── __init__.py
│   └── module.py
├── tests/
│   ├── __init__.py
│   └── test_module.py
├── README.md
├── LICENSE
├── pyproject.toml
```

Before publishing to the official PyPI, it's advisable to test your package on [TestPyPI](https://test.pypi.org/)

- Create an Account: Similar to PyPI, create an account on TestPyPI.
- Configure local machine

    ```bash
    poetry config repositories.testpypi https://test.pypi.org/legacy/ 
    poetry config pypi-token.testpypi your_testpypi_token_here
    ```

- Build and push to TestPypi `poetry publish --build testpypi`
- Go to <https://test.pypi.org/> and check if your project is there. You can optionally install the package as well

    ```bash
    pip install -i https://test.pypi.org/simple/ simple-robotics-python-utils
    ```

  - `-i` means `--index-url`. By default, pypi fetches packages from the official Pypi repo. However, we might need this when we have a private Pypi repo or TestPypi

Build and push to Pypi:

- Create an Account in Pypi and verify Your Email.
- Follow the Above steps for TestPypi and configure your API token

```bash
poetry config pypi-token.pypi your_pypi_token_here
```

- `poetry publish --build`

  - `poetry build` to build a package, which includes `dist` with `simple_robotics_python_utils-1.0.0-py3-none-any.whl` and `simple_robotics_python_utils-1.0.0.tar.gz`
  - If you see `There are 2 files ready for publishing. Build anyway? (yes/no) [no] yes`, just do `rm -rf dist`

## Pip

### Installation

- `--user`  installs the package in a local directory where the current user has permissions.

```bash
pip install --user <PKG>
```
