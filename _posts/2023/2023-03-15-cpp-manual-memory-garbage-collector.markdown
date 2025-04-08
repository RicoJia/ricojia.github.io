---
layout: post
title: C++ - Manual Memory Garbage Collector
date: '2023-01-26 13:19'
subtitle: Boehm-Demers-Weiser (BDW) Collector
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Boehm-Demers-Weiser Collector (BDW) Collector

BDW Collector is designed as a "drop-in" replacement in C for the default memory allocation functions (malloc, free, etc.). Developers can link against it with minimal changes to their existing code.

The collector scans regions of memory (such as the call stack, registers, and global data areas) for values that resemble pointers to allocated objects. If it finds such values, it marks the corresponding objects as reachable. It does not require precise knowledge about pointer locations, which is why it’s called "conservative."

### Scanning Process - Mark And Sweep

1. Gather roots (a root is a root pointer)
    - those from global variables, current stack frames, CPU registers, etc.
        - TODO: stack frame? What does it look like?
        - So the collector can programmatically gather the address of global variables? how does that work? 
2. Mark: 
    - The mark process is a recursive process that traverses through the tree. It will mark any object as a reference. It's a conservative collector meaning some non-pointer data are treated as pointer. 
3. Sweep:
    - Iterated through all objects on the heap. Ones that are not marked as reachable will be freed 

```c
function garbageCollect():
    // Phase 1: Mark
    rootSet = gatherRoots()  // Collect pointers from global variables, stacks, registers, etc.
    for each pointer in rootSet:
        mark(pointer)

    // Phase 2: Sweep
    for each object in heap:
        if object is not marked:
            free(object)
        else:
            object.marked = false  // Reset mark for next GC cycle

function mark(object):
    if object is null or object is already marked:
        return

    object.marked = true
    for each pointerField in object:
        mark(pointerField)
```
- The collector has knowledge about the boundaries of the heap
- Can the collector (c code) access call stack? 
    - It uses platform-specific APIs or symbols provided by the linker to access the boundary of the callstack
    - The collector iterates over the memory range of the callstack, examining each word to see if its value falls into the boundaries of allocated heap address. If so, it treats it as a pointer to a "reachable object" and mark true
- How does the collector grab all global variables?
    - Global and static variables are stored in designated sections of the program's memory (often the "data" and "BSS" segments). The garbage collector also scans these regions.