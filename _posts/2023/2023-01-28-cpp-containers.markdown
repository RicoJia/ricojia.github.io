---
layout: post
title: C++ - Container Operations
date: '2023-01-28 13:19'
subtitle: Vector, Map, Algorithms
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Vector

### Common Operations

- Append `vector2` to the end of `vector1`

```cpp
vector1.insert(vector1.end(), vector2.begin(), vector2.end());
```

## Map 

TODO

## Algorithms

### Nth Element

`nth_element(first, nth, last, comp)` makes sure:

1. At `nth` place of the container, the element is actually the `nth` as if the container were sorted.
2. Elements before the nth element has `comp(i, n) = False`, or without `comp`, they are smaller than the `nth` element

```cpp
// elements before the returned iterator has a value less than or equal to the value
std::nth_element(keypoints.begin(),keypoints.begin() + desired_features_num - 1, keypoints.end(),
    [](const KeyPoint& k1, const KeyPoint& k2){
        //descending 
        return k1.response > k2.response;
    }
);
```

### Partition

`std::partition(first, last, pred)` moves all elements that makes pred(item) **true** to  first iterator, and returns an iterator that points to the first item that makes pred(item) **false**. In combination wtih `std::nth_element`, we can partition and find the iterator that makes sure all elements smaller than or equal to the `nth` elements are before a returned iterator, `new_end`.

```cpp
// we might still have elements equal to the nth element. So, we use partition to find them
// std::partion moves items that satisfies the pred to before the iterator
auto new_end = std::partition(keypoints.begin() + desired_features_num, keypoints.end(), [&keypoints](const KeyPoint& k){k.response == (keypoints.begin() + desired_features_num - 1)->response});
```

## PMR: Polymorphic Memory Resources (C++17)

- Custom Allocators: Instead of having containers decide how memory is allocated, PMR allows you to supply a custom memory resource (allocator) that the container uses.
- Reusable Strategies: You can create memory resources that implement different allocation strategies, such as pooling, monotonic allocation, or synchronized (thread-safe) allocation, and then reuse them across multiple containers.
    - Memory Efficiency: PMR can help you pre-allocate a large chunk of memory (from a buffer) and then use it for many small allocations, which is useful in real-time or embedded systems.
    - Multiple PMR-based containers can share the same memory resource. 
- memory allocation strategies by using user-supplied memory resources.
