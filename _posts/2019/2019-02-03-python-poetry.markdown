---
layout: post
title: Python - Poetry
date: '2019-02-03 13:19'
subtitle: Poetry
comments: true
header-img: "img/home-bg-2015.jpg"
tags:
    - Python
---

## Introduction

Using Poetry is highly recommended because it simplifies dependency management and packaging, even when not using virtual environments. Environments without Poetry can install poetry packages because source distributions `sdist` and wheel files `.whl` are standard Python packages formats.

I'm using Docker and generally don't use `venv`. So I can disable venv by:

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