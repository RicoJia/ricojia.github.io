---
layout: post
title: Computer Vision - Collision Detection Using AABB Tree
date: '2025-08-04 13:19'
subtitle: AABB Tree
header-img: img/post-bg-kuaidi.jpg
tags:
    - Robotics
comments: true
---

## What is an AABB Tree, and How does it work?

AABB Tree is very efficient in all aspects in terms of game physics, including ray casting, point picking, region query, and, most importantly, generating a list of collider pairs that are potentially colliding (which is the main purpose of having a broadphase).

An **Axis-Aligned Bounding Box (AABB) Tree** is a bounding volume hierarchy (BVH). Each node stores a box whose sides are parallel to the coordinate axes and that is guaranteed to fully enclose all geometry in its subtree. The tree lets you reject large groups of geometry pairs cheaply: if two nodes' boxes don't overlap, none of their children can overlap either.

**Example — 4 robot links**

Suppose the robot has 4 links: `base`, `shoulder`, `elbow`, `wrist`.

```
         [root AABB  — encloses whole robot]
              /                  \
   [AABB: base+shoulder]    [AABB: elbow+wrist]
       /         \               /          \
  [base]    [shoulder]       [elbow]       [wrist]
```

To check all pairs:

1. Start at root pair `(root, root)` — trivially overlaps (same box), descend.
2. Compare `(base+shoulder, elbow+wrist)` — if these two sub-tree boxes don't overlap, **all 4 inter-group pairs are rejected in one test**.
3. If they do overlap, descend further: compare `(base, elbow)`, `(base, wrist)`, `(shoulder, elbow)`, `(shoulder, wrist)`.
4. Pairs within the same subtree (`base↔shoulder`, `elbow↔wrist`) are also checked recursively.

Result: instead of naively checking all $\binom{N}{2}$ pairs, most are pruned at internal nodes → **O(n log n)** on average.

**Dynamic AABB Tree** (what FCL uses here): the tree is kept *incrementally updated* as objects move. After `update()` calls `computeAABB()` on each FCL object, the manager re-fits the affected branches without rebuilding from scratch, keeping updates O(log n).

- **Narrowphase distance**: after the broadphase culls candidates down to overlapping AABB pairs, FCL's narrowphase `fcl::distance()` computes the *exact* minimum separation. For meshes (stored as OBBRSS BVH trees), it recursively descends both trees simultaneously — at each node it checks whether the OBB/RSS node pair can possibly be closer than the current best distance; if not, the whole subtree is pruned. It terminates at leaf triangles and runs a triangle-to-triangle distance computation. For primitives (sphere, box, cylinder) it uses closed-form analytic formulas instead.
- **Relationship to ROS / Gazebo**: MoveIt (ROS) uses FCL for the exact same purpose via `collision_detection::CollisionWorldFCL` — `CollisionMonitor` is essentially a real-time-control-friendly version of MoveIt's self-collision checker without the planning overhead. Gazebo is different: it uses ODE or Bullet as a *physics simulator* that resolves contacts with forces/constraints at runtime; it is not used for pre-emptive distance monitoring in a control loop.

## Overview

Dynamic AABB tree is a type of broadphase that is borderless, meaning that it does not impose a strict border limit as some other broadphases do, such as explicit grid and implicit grid.

---

## Reference

<https://allenchou.net/2014/02/game-physics-broadphase-dynamic-aabb-tree/>
