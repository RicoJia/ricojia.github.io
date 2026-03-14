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

## What is an AABB Tree?

An **Axis-Aligned Bounding Box (AABB) Tree** is a bounding volume hierarchy (BVH) where each node stores a box whose sides are parallel to the coordinate axes, guaranteed to fully enclose all geometry in its subtree.

It is efficient for game physics and robotics tasks including ray casting, point picking, region queries, and — most importantly — generating the candidate list of colliding pairs (the **broadphase**).

**Key pruning idea**: if two nodes' boxes don't overlap, *none* of their children can overlap either, because every child is contained inside its parent's box. This lets the algorithm reject large groups of pairs in a single O(1) test, giving **O(n log n)** average complexity vs. O(n²) brute force.

---

## Broadphase: Detecting Candidate Pairs

### The 4-Link Robot Example

Suppose a robot has 4 links: `base`, `shoulder`, `elbow`, `wrist`. Each link has an oriented geometry, but we wrap it in an axis-aligned bounding box (AABB):

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/GhvVhwcN/Screenshot-from-2026-03-14-12-20-26.png" height="300" alt=""/>
        <figcaption>Robot links and their AABBs</figcaption>
    </figure>
</p>
</div>

The tree groups links spatially:

```
         [root AABB  — encloses whole robot]
              /                  \
   [AABB: base+shoulder]    [AABB: elbow+wrist]
       /         \               /          \
  [base]    [shoulder]       [elbow]       [wrist]
```

The broadphase rule is simple:

```
if AABB_A overlaps AABB_B  →  possible collision → pass to narrowphase
else                        →  definitely no collision → prune
```

### Pair Traversal

Starting at `(root, root)`, the traversal descends recursively:

1. `(root, root)` — trivially overlaps, descend.
2. `(base+shoulder, elbow+wrist)` — if these don't overlap, **all 4 cross-group pairs are pruned in one test**; if they do, descend to `(base, elbow)`, `(base, wrist)`, `(shoulder, elbow)`, `(shoulder, wrist)`.
3. Within-subtree pairs — `(base, shoulder)` and `(elbow, wrist)` — are also checked recursively.

Two important notes:

- **Adjacent links** (e.g. `elbow` and `wrist`) will overlap at their shared joint — this is expected. They are filtered out by the **Allowed Collision Matrix (ACM)** after broadphase, not by the tree.
- **Non-adjacent links in the same subtree** that overlap *will* be captured. The tree has no topology-based skip; only the ACM decides which pairs to ultimately ignore.

---

## Tree Construction

Construction uses a **top-down, recursive median-split** strategy:

1. Compute a tight AABB around each object → these become leaf nodes.
2. Find the axis (x, y, or z) with the **largest spread** of object centers.
3. Sort objects by center along that axis; split at the median into left and right groups.
4. Recurse on each group until every node holds exactly one object.
5. On the way back up, set each internal node's AABB to the **union** of its children's AABBs.

**Example — 4 links along x:**

| Link | x-center |
|------|----------|
| base | 0.0 |
| shoulder | 0.5 |
| elbow | 1.2 |
| wrist | 1.8 |

Median split → left: `{base, shoulder}`, right: `{elbow, wrist}`. Recurse to leaves, then fit parent AABBs bottom-up:

```
         [root AABB = union of all four]
              /                  \
   [AABB: union(base,shoulder)]   [AABB: union(elbow,wrist)]
       /         \                    /            \
  [base]    [shoulder]           [elbow]          [wrist]
```

---

## Dynamic Updates: Joint State Changes

When a joint angle changes, forward kinematics moves one or more links in the chain. The tree **topology stays fixed**; only AABBs are updated.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/mr4Hf9dt/Screenshot-from-2026-03-14-12-40-55.png" height="300" alt=""/>
        <figcaption>After a joint move — wrist and base now overlap</figcaption>
    </figure>
</p>
</div>

**Update sequence:**

1. Joint angles change → FK recomputes link transforms.
2. Collision object transforms are updated.
3. Affected **leaf AABBs** are recomputed around the new geometry positions.
4. **Ancestors are refitted bottom-up**: each internal node's AABB becomes the union of its children's updated AABBs.

In FCL's Dynamic AABB Tree, `update(collision_object)` handles this by removing the leaf, recomputing its AABB, and reinserting it at an optimal position — the topology may change slightly to keep the tree balanced and tight.

**Cost**: $O(k \log n)$ for $k$ moved links, vs. $O(n \log n)$ for a full rebuild.

```
Joint changes  →  recompute leaf AABBs  →  refit ancestors bottom-up
     O(k)               O(k)                      O(k log n)
```

**Cross-subtree detection**: `base` and `wrist` don't need to be in the same subtree for their collision to be found. After the move, the broadphase traversal will test `overlap(AABB_base, AABB_wrist)` when it descends all cross-subtree pairs — if they overlap, `base ↔ wrist` is reported regardless of tree membership.

---

## Full Pipeline (Pseudocode)

### 1. Tree Construction

```python
def build_aabb_tree(objects):
    """Top-down recursive median split."""
    if len(objects) == 1:
        return LeafNode(aabb=compute_aabb(objects[0]), obj=objects[0])

    axis = axis_of_largest_spread(objects)          # x, y, or z
    objects.sort(key=lambda o: center(o)[axis])
    mid = len(objects) // 2

    left  = build_aabb_tree(objects[:mid])
    right = build_aabb_tree(objects[mid:])

    return InternalNode(
        aabb  = union(left.aabb, right.aabb),       # bottom-up fit
        left  = left,
        right = right,
    )
```

### 2. Broadphase — AABB Overlap Traversal

```python
def broadphase(nodeA, nodeB, candidates):
    if not overlap(nodeA.aabb, nodeB.aabb):
        return                                      # prune entire subtree pair

    if nodeA.is_leaf and nodeB.is_leaf:
        candidates.append((nodeA.obj, nodeB.obj))
        return

    # descend into the larger node's children to keep pairs balanced
    if nodeA.is_leaf or (not nodeB.is_leaf and nodeB.aabb.volume > nodeA.aabb.volume):
        broadphase(nodeA, nodeB.left,  candidates)
        broadphase(nodeA, nodeB.right, candidates)
    else:
        broadphase(nodeA.left,  nodeB, candidates)
        broadphase(nodeA.right, nodeB, candidates)

candidates = []
broadphase(root, root, candidates)                  # self-collision check
```

The traversal is **top-down with pruning**: `overlap()` at each level is a gate — if parent boxes don't overlap, no children are ever visited. Tracing the 4-link example:

```
broadphase(root, root)                        ← same box → descend
  broadphase(base+shoulder, root)             ← overlap → descend
    broadphase(base+shoulder, base+shoulder)  ← self → descend
      broadphase(base, shoulder)              ← leaves → add pair
      broadphase(shoulder, base)              ← duplicate, skipped elsewhere
    broadphase(base+shoulder, elbow+wrist)    ← NO overlap → prune all 4 cross pairs instantly
                                              ← overlap → (base,elbow),(base,wrist),(shoulder,elbow),(shoulder,wrist)
  broadphase(elbow+wrist, root)               ← similar for right subtree
```

### 3. Narrowphase — Exact Distance / Collision Test

```python
def narrowphase(candidates, acm, threshold=0.0):
    results = []
    for (objA, objB) in candidates:
        if acm.is_allowed(objA, objB):              # skip adjacent/ignored pairs
            continue

        if objA.is_primitive and objB.is_primitive:
            dist = analytic_distance(objA, objB)    # closed-form (sphere, box, …)
        else:
            dist = bvh_distance(objA.mesh, objB.mesh)  # recursive OBB/RSS descent

        if dist <= threshold:
            results.append(CollisionPair(objA, objB, dist))
    return results
```

`bvh_distance` recursively descends both mesh BVH trees simultaneously, pruning subtree pairs whose minimum possible distance already exceeds the current best, and running triangle-to-triangle tests at the leaves.

---

## Mesh Collision Detection (Narrowphase Leaves)

At the leaves of a mesh BVH, collision reduces to two triangle-level primitives.

### Triangle–Triangle Intersection (Möller's Algorithm)

Uses the **Separating Axis Theorem (SAT)** on 11 candidate axes: the 2 triangle normals and the 9 edge×edge cross-products.

```python
def triangles_intersect(T1, T2):
    # T1 = (v0,v1,v2), T2 = (u0,u1,u2)

    # Test 1: plane of T1 separates T2?
    n1  = cross(v1-v0, v2-v0)
    d_u = [dot(n1, u-v0) for u in T2]
    if all_same_sign_nonzero(d_u):
        return False                     # T2 entirely on one side

    # Test 2: plane of T2 separates T1?
    n2  = cross(u1-u0, u2-u0)
    d_v = [dot(n2, v-u0) for v in T1]
    if all_same_sign_nonzero(d_v):
        return False

    # Test 3: 9 edge×edge separating axes
    for e1 in edges(T1):
        for e2 in edges(T2):
            axis = cross(e1, e2)
            if near_zero(axis): continue             # parallel edges
            p1 = [dot(axis, v) for v in T1]
            p2 = [dot(axis, u) for u in T2]
            if max(p1) < min(p2) or max(p2) < min(p1):
                return False                         # separating axis found

    return True                                      # no separator → intersecting
```

### Triangle–Triangle Minimum Distance

When the triangles don't intersect, the closest points lie on their boundaries. All feature pairs are checked:

```python
def triangle_triangle_distance(T1, T2):
    if triangles_intersect(T1, T2):
        return 0.0, None, None

    best = inf
    # 9 edge–edge pairs
    for e1 in edges(T1):
        for e2 in edges(T2):
            d, p, q = segment_segment_distance(e1, e2)
            if d < best: best, best_p, best_q = d, p, q

    # 3 vertices of T1 vs face of T2, and vice-versa
    for v in T1.vertices:
        d, p, q = point_triangle_distance(v, T2)
        if d < best: best, best_p, best_q = d, p, q
    for v in T2.vertices:
        d, p, q = point_triangle_distance(v, T1)
        if d < best: best, best_p, best_q = d, p, q

    return best, best_p, best_q
```

---

## In the ROS World

[FCL](https://github.com/flexible-collision-library/fcl) supports five proximity query types on triangle mesh models:

- **Collision detection**: do the models overlap? Which triangles?
- **Distance computation**: minimum separation distance and closest point pair.
- **Tolerance verification**: are two models closer or farther than a threshold?
- **Continuous collision detection**: do moving models overlap during the motion?
- **Contact information**: contact normals and contact points.

**MoveIt** (ROS) uses FCL for self-collision checking via `collision_detection::CollisionWorldFCL`. **Gazebo** uses ODE or Bullet as a full physics simulator that resolves contacts with forces at runtime — a different use case from pre-emptive distance monitoring in a control loop.

---

## Reference

<https://allenchou.net/2014/02/game-physics-broadphase-dynamic-aabb-tree/>
, including ray casting, point picking, region query, and, most importantly, generating a list of collider pairs that are potentially colliding (which is the main purpose of having a broadphase).

An **Axis-Aligned Bounding Box (AABB) Tree** is a bounding volume hierarchy (BVH). Each node stores a box whose sides are parallel to the coordinate axes and that is guaranteed to fully enclose all geometry in its subtree.

Naively, you can check all $2^n$ pairs of the bounding boxes:

Key idea: If two nodes' boxes don't overlap, none of their children can overlap — allowing large groups of geometry pairs to be rejected **cheaply**:

1. Compare bounding boxes at the root level first
2. If sub-tree boxes don't overlap → prune all child pairs in one test
3. Only descend into subtrees where boxes do overlap
4. Leaf nodes represent actual geometry (links)

This results in O(n log n) average complexity vs. O(n²) brute force.
