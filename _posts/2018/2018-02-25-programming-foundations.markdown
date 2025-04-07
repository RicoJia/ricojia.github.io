---
layout: post
title: Foundamental Structure of Programs
date: '2024-05-01 13:19'
subtitle: Callstack
comments: true
tags:
    - Linux
---

## Callstack

A callstack is a stack of frames, where a frame represents an active call. Each frame has: return address (where the function returns after it completes), local variables. 

E.g., if `function A -> B -> C`. it will look like:

```
[ Frame for function C ] <-- Top of the stack
[ Frame for function B ]
[ Frame for function A ] <-- Bottom (or base) of the stack
```