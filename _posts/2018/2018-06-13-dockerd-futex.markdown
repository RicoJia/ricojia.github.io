---
layout: post
title: Dockerd and Linux Futex
date: 2024-06-10 13:19
subtitle: How Docker uses containerd, and how Linux futex works
comments: true
tags:
  - Docker
  - Linux
---

### Containerd

**containerd** is the **low-level container runtime** that Docker uses under the hood. It manages image storage, pulling images, unpacking layers, and container lifecycle. For example, `docker pull ubuntu` actually goes through containerd.

```
Your command → docker CLI
            → dockerd (Docker daemon)
            → containerd
            → runc (actually runs containers)
```

containerd has its own image store — images are stored directly on containerd.

Docker is written in Go and uses goroutines to pull images in multiple threads. For `docker save myimage > myimage.tar`, it exports a Docker image into a tar archive:

1. Ask containerd for image data
2. Reconstruct layers
3. Serialize everything into tar
4. Stream it out

In contrast, `docker push → registry → docker pull`:

- Streams layers directly
- Avoids reconstructing the whole image tar
- Uses well-tested registry APIs

This is much less likely to hit deadlocks caused by futex contention in Go's goroutine scheduler.

### Futex

In Linux, a **futex** (fast user-space mutex) is a low-level kernel mechanism for building locks like mutexes. It can work across processes and is described as a "sleep/wake mechanism tied to a memory address."

- Most locking happens in **user space** (fast path)
- The kernel is only involved when contention happens (slow path)

**How it works:**

1. Thread tries to lock — checks a memory value (no kernel call yet)
2. If lock is free → take it
3. If locked → call `futex` syscall: "sleep until this memory changes"

**Why it's called _fast_:**

- Uncontended locks → no syscall
- Only on contention → kernel involvement

If the memory is shared (e.g., via `mmap`), futex works across processes. It is not a mutex itself, but the **primitive used to build mutexes**.
