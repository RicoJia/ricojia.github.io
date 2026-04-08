---
layout: post
title: Python - Enums, dataclass
date: 2019-01-15 13:19
subtitle: Enum, IntEnum
comments: true
header-img: img/post-bg-2015.jpg
tags:
  - Python
---

## `IntEnum`

- IntEnum's members are ints, while enum instance's members are its own class
    - `Enum: Base class for creating enumerated constants.`
    - `IntEnum: Base class for creating enumerated constants that are also subclasses of int.`
        ```
        class Shape(IntEnum):
            CIRCLE = 1
            SQUARE = 2

        class Color(Enum):
            RED = 1
            GREEN = 2

        Shape.CIRCLE == Color.RED
        >> False

        Shape.CIRCLE == 1
        >>True
        ```
    - Also, this would raise: `ValueError: invalid literal for int() with base 10: 'a'`
        ```
        class State(IntEnum):
            READY = 'a'
            IN_PROGRESS = 'b'
            FINISHED = 'c'
            FAILED = 'd'
        ```
    - Create `IntEnum` instance with data:
        ```
        self.motor_error_code = MotorErrorCode(self.motor_error_code)
        ```

---

## Dataclass

The `@dataclass(frozen=True)` decorator defines a class whose instances are simple data containers with automatically generated methods. By using `@dataclass`, Python creates common methods such as `__init__` (constructor), `__repr__`, and `__eq__`, eliminating the need to write boilerplate code manually. 

The `frozen=True` argument makes instances of the class immutable, meaning that once an object is created, its fields cannot be modified.

```python
@dataclass(frozen=True) 
class Config:
```