---
layout: post
title: C++ - Segfaults
date: 2024-04-12 13:19
subtitle: Out-of-bounds Iterator Access, Core Dump
comments: true
header-img: img/post-bg-alitrip.jpg
tags:
  - ROS
---

## Out-of-bounds Iterator Access

Out-of-bounds iterator access is an undefined behavior, which might give a segfault, or give a garbage value.

```cpp
#include <vector>
#include <iostream>

int main() {
    std::vector<int> v = {10, 20, 30};
    auto iter_first = v.begin();
    auto bad_iter = iter_first + 5;

    // Dereference — UB, likely a segfault
    std::cout << *bad_iter << "\n";
    return 0;
}
```

## Workflow - Coredump

On Linux, debugging a segfault usually means inspecting a **core dump**, which is a snapshot of a **process’s memory** at the moment it crashed. Whether a core dump is produced — and where it ends up — depends on a few layers of configuration.

`ulimit` controls **per-shell resource limits** for processes you start from that shell. In particular, `ulimit -c` sets the maximum size of a core dump; if it is `0`, no core dump will be generated at all. Setting it to `unlimited` enables core dump generation for that session and its child processes.

On most modern distributions, core dumps are captured by **systemd-coredump** rather than written directly to files. In this setup, crashes are recorded by systemd and accessed via `coredumpctl`, which can list crashes, show metadata, dump the core to a file, or launch `gdb` directly on the crashing binary.

Optionally, the kernel’s `core_pattern` can be changed to bypass systemd and write core dumps as plain files instead. This is a **system-wide setting** and should be used with care, but can be useful in environments without systemd (e.g. minimal containers) or when you explicitly want a local core file.

```
ulimit -c unlimited

// systemd-coredump
// If you see something piped into systemd (often contains `systemd-coredump`), then `coredumpctl` is the right tool.
cat /proc/sys/kernel/core_pattern


// ### (Optional) Check coredump storage policy
cat /etc/systemd/coredump.conf

// dump core into a plain file. 
// It affects the whole machine**, not just your ROS 2 node.
sudo sh -c 'echo "core_backend_test.%p" > /proc/sys/kernel/core_pattern'
```

- `coredumpctl` is a command-line tool that comes with **systemd**. On many distros, systemd captures crashes and stores the “core dump” in the journal or on disk. `coredumpctl` lets you:
 	- list recorded crashes.  dump the core to a file, launch gdb directly on them
