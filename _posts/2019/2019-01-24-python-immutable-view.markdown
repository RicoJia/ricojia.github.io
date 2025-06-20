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

Python has mutable and immutable objects:

Mutables:

- bytearray
- list
- ...

Immutables:

- string
- bytes
- ...

```python
buf = bytearray(b"hello world")
mv = memoryview(buf)
mv2 = mv[6:11]        # refers to the same bytes as buf[6:11]
mv2.tobytes()         # => b"world", but buf isn’t duplicated in memory until you call tobytes()
```

- **Bytes is immutable, meaning once created, its contents cannot be changed. So any slice of it will be a new object**
- By default, **slice of a mutable object is also a new object**

## Memory View

It returns the memory of the underlying object. To avoid copies, use `memoryview`:

```python
buf = bytearray(1024)
view = memoryview(buf)[100:]   # no copy
n = sock.recv_into(view)       # writes *directly* into buf[100:100+n]
```

- The `memoryview` object of a mutable object is writable, otherwise it'd be read-only for an immutable

### Advanced buffer‐protocol features

You can cast a `memoryview` object to other element types, like raw bytes as 32-bit ints
