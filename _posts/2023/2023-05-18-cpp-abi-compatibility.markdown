---
layout: post
title: C++ - ABI Compatibility
date: '2023-04-09 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-infinity.jpg"
tags:
    - C++
---

## ABI Compatibility: Introduction and Motivating Example

ABI stands for Application Binary Interface. Two versions of the same library might be API-Compatible (you can recompile your code), but they are not ABI-compatible (you cannot just swap the binary `.so` without recompiling). Here's a motivating example:

We have two versions of the same data schema, and some fields are swapped. In serialization / deserialization, it's common to make assumptions about the order of fields. So such a order change would likeliy not cause compilation issue, but will cause data interpretation issue.

```cpp
// Version 1:
struct DataPacket{
    int value1;
    int value2;
};

// Version 2:
struct DataPacket{
    int value2;
    int value1;
};
```