---
layout: post
title: Python - Messaging Tools
date: 2019-03-14 13:19
subtitle: zeromq
comments: true
header-img: img/post-bg-2015.jpg
tags:
  - Python
---
## ZeroMQ

### Blocking vs Poll

Block could get stuck here :  - **Worker thread, as in our example:** Ctrl+C goes to the main thread. It does **not automatically interrupt the worker’s `recv_multipart()`**, so that worker can remain blocked.
```
while running:
    frames = socket.recv_multipart()  # Wait here until a message arrives
```

ctrl-C goes to the main thread. So if this is running on the main thread, `recv` will be interrupted. Otherwise, if this runs on another thread, the worker thread can remain blocked. 

With a timeout-based poll:

```python
while running:
    events = dict(poller.poll(100))
    if socket in events:
        frames = socket.recv_multipart()
```

Here, If data arrives, wake up immediately. Otherwise, sleep for up to 100 milliseconds, **IDLE,** then return. 

This one is busy polling, which could consume some CPU:

```python
while running:
    events = dict(poller.poll(0))  # Return immediately; loop again
```