---
layout: post
title: "[ML] K-Nearest-Neighbor Using Heap With CUDA"
date: 2025-01-09 13:19
subtitle:
header-img: img/post-bg-o.jpg
tags:
  - Machine-Learning
comments: true
---
---

## KNN (Heap-Based Implementation)

**Goal:**  For a query point qqq, find its kkk nearest neighbors from a set of NNN points.

---

### Algorithm Idea

We use a **max-heap of size k**. Why a max-heap?

- The heap always stores the **current best k nearest neighbors**.
- The **root (first element)** of the heap is the **largest distance among those kkk**. That means the root is the **worst neighbor in the current top-k set**.
- **Any new candidate must beat (be smaller than) the root to enter the heap.** -> This guarantees k best candidates

### Step-by-Step Algorithm

For each query point `p`:

1. Initialize a max-heap of size k, filled with $\infty$

2. For each point p ​:

    - Compute squared distance:
  $$
  d_i^2 = (q - p_i)^2
  $$
    - If:
  $$
  d_i^2 < \text{heap}[0]
  $$

        then:

        - Replace the root with $d_i^2$​

        - Heapify-down to restore max-heap property

1. Return the first element as the final result

---

## Example

Let’s use a simple 1D example. Database points:

```
P = [1, 3, 6, 8, 10]
```

Query:

```
q = 7  
k = 2
```

 Compute squared distances

| Point | Distance | Squared |
| ----- | -------- | ------- |
| 1     | 6        | 36      |
| 3     | 4        | 16      |
| 6     | 1        | 1       |
| 8     | 1        | 1       |
| 10    | 3        | 9       |

- Initialize heap (size 2), `heap = [∞, ∞]`
- Insert 36:
 	- `heap = [36, ∞]`
 	- Heapify (max-heap): `heap = [∞, 36]`
- Insert 16:
 	- Since 16 < ∞ → replace root: `heap = [16, 36]`
 	- Heapify: `heap = [36, 16]`. Root (worst among them) = 36
- Insert 1:
 	- 1 < 36 → replace root: `heap = [1, 16]`
 	- Heapify: `heap = [16, 1]`
- Insert another 1:
 	- 1 < 16 → replace root: `heap = [1, 1]`
 	- Heapify → no change
- Insert 9:
 	- 9 < 1? No → ignore.
- Final result:
 	- Heap contains: `[1, 1]`
 	- Corresponding points: 6 and 8

## Computational Complexity

For one query point:

$$
O(Nlogk)
$$

For M query points:

$$
O(NMlogk)
$$

This is efficient when:

$$
k \ll N
$$
