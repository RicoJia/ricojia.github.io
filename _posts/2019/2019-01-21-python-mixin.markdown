---
layout: post
title: Python - Python Mixin Classes
date: '2019-01-21 13:19'
subtitle: MRO
comments: true
header-img: "img/post-bg-2015.jpg"
tags:
    - Python
---

## Mixins

A mixin is a small, focused class that provides methods to be “mixed in” to other classes via multiple inheritance. Mixins let you bundle reusable behaviors and add them to any class, keeping your code DRY and your class hierarchies clean.

- They are usually not a standalone base class, so mixins generally don’t define their own `__init__` or **hold persistent attributes**; they assume the main class provides any needed state.

- A mixin implements **one cohesive piece of functionality** (e.g. logging, serialization, validation).

- Multiple Inheritance: You inherit from one “real” base class (or more) plus any number of mixin classes:
  - Order matters: Python’s Method Resolution Order (MRO) determines which method is called when names collide. Always put your mixins after the primary base class in the inheritance list.

```python
class LoggingMixin:
    def log(self, message: str) -> None:
        print(f"[{self.__class__.__name__}] {message}")

import json

class JsonSerializableMixin:
    def to_json(self) -> str:
        # Assume `self.__dict__` holds only serializable values
        return json.dumps(self.__dict__)

# Combine Them with a “Real” Class
class User(LoggingMixin, JsonSerializableMixin):
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

# Usage
user = User("Alice", "alice@example.com")
user.log("Created user")           # → [User] Created user
print(user.to_json())              # → {"name": "Alice", "email": "alice@example.com"}
```

## MRO TODO
