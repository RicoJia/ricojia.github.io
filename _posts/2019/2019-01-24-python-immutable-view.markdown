---
layout: post
title: Python - Immutable And Memory View
date: '2019-01-21 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-2015.jpg"
tags:
    - Python
---

## Mutable vs Immutable Objects

Python objects are either mutable or immutable:

| Mutable | Immutable |
|---------|-----------|
| `bytearray`, `list`, `dict`, `set` | `str`, `bytes`, `int`, `tuple` |

Key slicing behaviour:

- A slice of an **immutable** object (e.g. `bytes`) always creates a **new copy**.
- A slice of a **mutable** object also creates a new copy by default — unless you use `memoryview`.

## Memory View

`memoryview` exposes the underlying buffer of an object without copying it. This is useful for working with large binary data efficiently.

```python
buf = bytearray(b"hello world")
mv = memoryview(buf)
mv2 = mv[6:11]        # zero-copy slice — still refers to buf[6:11]
mv2.tobytes()         # => b"world" (copy only happens here)
```

A common use case is writing directly into a pre-allocated buffer:

```python
buf = bytearray(1024)
view = memoryview(buf)[100:]   # no copy
n = sock.recv_into(view)       # writes directly into buf[100:100+n]
```

- A `memoryview` of a **mutable** object is **writable**.
- A `memoryview` of an **immutable** object is **read-only**.

### Advanced: Casting to Other Types

You can reinterpret the raw bytes as a different element type using `.cast()`. For example, treating raw bytes as 32-bit integers:

```python
buf = bytearray(b'\x01\x00\x00\x00\x02\x00\x00\x00')
mv = memoryview(buf).cast('I')  # unsigned 32-bit ints
list(mv)  # => [1, 2]
```
