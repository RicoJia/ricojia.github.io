---
layout: post
title: Bash Process Group
date: '2023-06-16 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-infinity.jpg"
tags:
    - Linux
---
### What is a process group?

In Unix/Linux terms, a **process group** is a way the OS groups related processes so they can be signaled and managed as a unit. Each group is identified by a **process group ID (PGID)**.

```bash
kill -TERM -12345
```

You’re not killing PID `12345` — the leading minus means send the signal to the entire process group with PGID `12345`.

Process groups are used mainly for **job control** and **terminal/shell management**.

Example:

```bash
ros2 run my_pkg my_node | tee output.log
```

That single pipeline creates multiple processes:

- `bash`
- `ros2`
- your node process
- `tee`
  - `tee` reads standard input and writes it both to standard output and to one or more files — effectively "tees" the stream so you can view output and save it at the same time.

When you press **Ctrl+C**, the terminal sends `SIGINT` to the foreground process group, so all of those processes receive the signal and stop together.

## Process group vs parent/child (important distinction)

- **Parent/child**: who started whom.
- **Process group**: which processes should be controlled together.

These are orthogonal concepts. For example:

- One parent, many children, all in the **same process group**.
- One parent with children in _different_ groups.

## Creating a new process group

```python
subprocess.Popen(
    cmd,
    start_new_session=True
)
```

On Linux this uses `setsid()` under the hood and creates a new session and a new process group; the child becomes `PGID = PID`.

Avoid patterns that create zombies or make process trees hard to manage:

- **Don’t** use `pkill -9` inside `_start_processes()` (this can create zombies and kill unrelated processes).
- **Prefer** launching without `bash -c` when possible (shell wrappers complicate process trees).

Typical process tree example: `subprocess.Popen()` → `bash` → `ros2` → actual node process. If you kill the node (a grandchild), the parent may become a defunct/zombie process if it doesn’t reap its children properly.
