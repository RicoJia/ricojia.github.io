---
layout: post
title: C++ [Container 3] Container Adaptors
date: '2023-01-29 13:19'
subtitle: std::queue, std::stack, std::priority_queue
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

`std::queue`, `std::stack`, `std::priority_queue` are NOT sequence containers, instead, they are wrappers around `std::deque`, and `std::vector` to provide a restricted interface. (FIFO for queue, LIFO for stack). The common interface we examine are:

- `push`, `pop`
- `top`, `front`
- No iterators or allow traversals.

### std::queue vs std::deque

`std::queue<LoopCandidate> loop_candidate_queue_;` vs `std::deque`

- **`std::queue<T>`** is a **restricted interface** (adapter). By default it’s implemented on top of **`std::deque<T>`**.
  - Ops you get: `push`, `emplace`, `front`, `back`, `pop`, `empty`, `size`
  - You _can’t_ iterate, index, erase arbitrary elements, etc.
  - It clearly communicates intent: _this is FIFO, don’t poke at it_.
- **`std::deque<T>`** is a **general-purpose container** (double-ended queue).
  - Ops you get: everything above plus iteration, `operator[]`, `insert/erase` (in middle), `push_front`, etc.
  - More flexible, but easier for callers (including future you) to accidentally rely on non-FIFO behavior.

### Performance (practical view)

- `std::queue<T>` **adds essentially no runtime overhead** vs using the underlying container directly; it’s just a thin wrapper.
- Both `std::deque` and a `std::queue`-over-`deque` give **amortized O(1)** push/pop at ends.
 	- amortized (on average)
- `std::queue` won’t magically be faster; it just prevents misuse.
