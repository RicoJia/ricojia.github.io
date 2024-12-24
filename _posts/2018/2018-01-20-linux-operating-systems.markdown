---
layout: post
title: Linux - Operating System
date: '2018-01-20 13:19'
subtitle: Paging, Concurrency
comments: true
tags:
    - Linux
---

## Paging

A page of memory is the smallest fixed-length block of virtual memory managed by OS. To use virtual memory, the OS needs to transfer pages between the main memory and the secondary memory. Pages are stored in a page table. Each page has a virtual address called page number

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/98ac5a90-ff2b-4e1e-8227-5bf8eca0aa69" height="300" alt=""/>
        <figcaption><a href="">Source: </a></figcaption>
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