---
layout: post
title: Linux - Operating System
date: '2018-01-20 13:19'
subtitle: Paging, Concurrency, Time
comments: true
tags:
    - Linux
---

## Paging

A page of memory is the smallest fixed-length block of virtual memory managed by OS. To use virtual memory, the OS needs to transfer pages between the main memory and the secondary memory. Pages are stored in a page table. Each page has a virtual address called page number

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/98ac5a90-ff2b-4e1e-8227-5bf8eca0aa69" height="500" alt=""/>
    </figure>
</p>
</div> 

Paging Life Cycle:
1. If the page table finds a page in RAM (virtual address lookup), virtual memory mapping is established directly. 
2. When the virtual address lookup fails, a page fault is generated. Then page fault exception handler writes a victim page in cache to disk, and load the target page from disk. Meanwhile the page table updates the virtual address mapping
    - In paging, writing from RAM to disk is faster in chunks than in individual bits. 


## Concurrency 

Event Driven frameworks works by handling events one at a time:

- `select(fd_set *readfds, fd_set *writefds, fd_set *errorfds, struct timeval *timeout)`
    1. takes in list of fds to check for reading, writing, or error condition pending.
    1. if no fd is ready for any of the above, it will poll, and block the thread until at least one fd is ready, within the timeout

- Each thread in Python occupies 8MB of virtual memory. However, only a small fraction of the virtual memory is mapped to the actual memory

## Time

- Wall (Real) time: Clock-time elapsed between program start and finish. It includes process and waits. `Wall time = blocked time + ready time + CPU running time`
    - Blocked time: Periods when the process is not runnable because it is waiting for:
        - I/O calls like file read-write, internet comm
        - Synchronization / signalling; Resources like mutex, condition variables, futexes, semaphores
        - Other resources: timers, sleep
    - Ready time: time the process sits in the run queue ready to execute, but the scheduler has not yet dispatched it to a CPU.
    - Running time: Time the process is actually executing on a CPU core. `Running time =  User time + Kernel (or Sys) time`
        - User time = time spent in the user space. E.g., for loops
        - Kernel time: time spent in kernel mode, 
            - System-call handlers (read, write, open, mmap, etc.)
            - Context-switch and scheduler overhead
        - Note: if there are multiple sub-processes created by fork, because the `total kernel time = sum (sub_process_times)`
    - CPU Bound and I/O Bound: 
        - CPU Bound: Most wall time is spent in CPU running time (user + system). Adding more CPU cores or optimising computation can yield speed-ups.
        - I/O Bound: A large fraction of wall time is in blocked time. Pure CPU parallelism helps little; throughput is limited by I/O latency or bandwidth. (Concurrency can still overlap waits, but total wall time is capped by the slowest I/O path.)
