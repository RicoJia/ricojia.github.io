---
layout: post
title: Robotics - Point Cloud Processing and KNN Problem
date: '2024-04-07 13:19'
subtitle: Bruteforce KNN Search, KD Tree, OctoTree
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
---

## Overview

Here is a [post about the LiDAR Working Principle and different types of LiDARs](https://ricojia.github.io/2024/10/22/robotics-3D-Lidar-Selection/).

One lidar model is the Range-Azimuth-Elevation model (RAE)

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/f6b078a2-9ff7-49e0-8cab-a841b595075e" height="300" alt=""/>
       </figure>
    </p>
</div>

Its cartesian coordinates are:

$$
\begin{gather*}
\begin{aligned}
& [r cos(E) cos(A), r cos(E) sin(A), r sin(E)]
\end{aligned}
\end{gather*}
$$

If given $[x,y,z]$ coordinates, it's easy to go the other way, too:

$$
\begin{gather*}
\begin{aligned}
& r = \sqrt{x^2 + y^2 + z^2}
\\ &
A = arctan2(\frac{y}{x})
\\ &
E = arctan2(\frac{z}{r})
\end{aligned}
\end{gather*}
$$

Some point clouds have meta data such as reflectance (intensity), RGB, etc. Reflectance can be used for ground detection, lane detection. Based on the above, LiDAR manufacturers transmit **packets** (usually) through UDP after measurements. Velodyne HDL-64S3 is a 64 line LiDAR. At each azimuth angle azimuth, (rotational position in the chart), a message contains on laser block that contains 32 lines (along the elevation angle). Its packet looks like:

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/cc966972-8fc2-4129-accb-358e20f506fd" height="300" alt=""/>
       </figure>
    </p>
</div>

- So, to find the `[x,y,z]` of a single lidar point,
  - `elevation_angle = 32*block_id + array_index`
  - `azimuth = rotational_position * 0.01 deg / 180 deg * pi` (Velodyne lidar azimuth increments in 0.01 deg)
  - `range = array[array_index]`
- In total, we need `2 + 2 + 32 *3 = 100 bytes` to represent the LiDAR Data. If we use the `pcl::PointCloud`, each point would be [x, y, z, intensity], and that's `32 * (4 + 4 + 4 + 1) = 416bytes`. **So packets are compressed LiDAR data.**

Aside from 3D Point Cloud, another 3D representation is a Surface Element Map (Surfel). A surface element is a small 3D patch that contains:

- `x,y,z`
- Surface normal vector
- Color / texture
- Size / Radius
- Confidence

## Introduction: Point Cloud Query

In SLAM, we often need to query a point cloud map about the occupancy of points. In effect, this is a query function: `bool is_there_a_point(point)`. Or, we try to find the nearest neighbor of that point. This is a mapping problem. Before jumping into the classical algorithms, we note that this can be learned by neaural networks (NN) **without labelling**. NN can also learn the implicit relationships in the pointcloud, like connectivity of the points; NN can also learn a signed-distance-field (SDF) as well. Note that **an NN for querying one point cloud map may not be usable for another point cloud**. But if we have a huge point cloud that doesn't change for a good amount of time, NN can generate a good querying result:

(See how smooth the reconstructed pointcloud is?)

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="" height="300" alt=""/>
    </figure>
</p>
</div>

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/f99cb0fb-d859-458d-bb1b-d19552df802b" height="300" alt=""/>
            <figcaption><a href="https://arxiv.org/pdf/1812.03828">Source: Occupancy Networks: Learning 3D Reconstruction in Function Space </a></figcaption>
       </figure>
    </p>
</div>

Some other reaserchers focus on finding descriptors of point clouds. For SLAM, a tiny bit of performance improvement can create a huge difference! Therefore, we must consider parallelism in implementation.

Now, let's get into the conventional methods. Most There are 2 popular ways for point cloud querying in SLAM

1. Point cloud is spatial. We can store them in spatial representations for faster indexing, like in a pixel / voxel grid.
2. We can store a point cloud in a tree, a binary-search tree /KD tree, or a quad-tree or octo tree.
    - Other trees like B-Tree are suitable for large static point clouds. In SLAM, we need to rapidly query in different point clouds.

## Brute Force KNN Search

Brute-Force KNN is very easy to parallelize, and it could be more efficient than more sophisticated algorithms.

Brute force KNN is:

1. Given a point, search through all points to find their distances. Find the K shortest distances

For K-Nearest-Neighbor search, we need an extra sorting process to find the k shortest distances. Of course, we are able to achieve 100% recall and precision using the brute force search

## One-Pass Grid Search Method

A faster method is to put point cloud into pixel / voxels. Here is how it's done:

1. For a given point `p`, calculate the coordinate `c` of with the known resolution r
    - Note that adjacent points may have the same coordinate.
2. Hash the coordinate into an integer
    - We use the **Locality-Sensitive-Hashing** function in "Optimized Hashing Function for Collision Checks". [Reference](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf)
    - `hash(x,y,z) = ( x p1 xor y p2 xor z p3) mod n`, where p1, p2, p3 are large prime numbers, in our case 73856093, 19349663, 83492791.
3. Store the point into an `unordered_map` into an array with hash.

During Search:

1. Calculate the hash of the query.
2. Fetch all points of the pixel / voxels under the same hash, and their **neighbors** (see below).
3. Brute force search through these points, find K nearest neighbors.

For 2D grids, we can have center (0-neighbor) 4-neighbors, 8 Neighbors.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/d90b52d6-32d6-472e-940f-e1a7a3737cc9" height="200" alt=""/>
    </figure>
</p>
</div>

For 3D voxel grids, there are 6-neighbors, 18-neighbors, 26-neighbors.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/ac1d1bde-29ca-4e81-8f4b-760a40812b93" height="300" alt=""/>
       </figure>
    </p>
</div>

## KD Tree

In a KD Tree, there are 2 branches of a non-leaf node. A Leaf node represents an actual point, a non-leaf node doesn't represent an actual point, but it just helps direct query searches to the right leaf node. A branch represents a "split" (or a "hyper plane") at the specified dimension. KD-tree can be easily implemented in the recursive form. **One thing to note is recursion may cause stack overflow.** Using for-loops would not.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/e5433e1e-ff23-42ba-8804-3a06bf5cb806" height="300" alt=""/>
       </figure>
    </p>
</div>

Tree Building:

1. Input: a group of points `[x, y, z]`
2. If the group is empty, return
3. If the group contains only 1 point, insert a new leaf node into the tree, return
4. Otherwise, the group has multiple points. Calculate the mean and variance of these points across every dimension.
5. Find the dimension with the largest variance. That will be our split dimension. The mean will be the split threshold.
6. Partition the point group into two, the group below split threshold is inserted into the left child node, the other is inserted into the right child node.
7. For each child node, repeat from step 1.

Tree searching:

The reason why tree searching is faster than brute force is because of tree-pruning. Tree pruning is: if the worst distance between my current candidates and the query is better than that between the query and a given split, we don't need to look into that split.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/40bee3b4-2355-43d6-9e0a-101875600f12" height="300" alt=""/>
       </figure>
    </p>
</div>

1. Input: current tree node, query point p, K
2. Output: list of K nearest neighbors
3. For the current node `nc`:
    1. If `nc` is a leaf,
        1. calculate its distance to p, `d_cp`
        2. If `d_cp` < max(current_k_neighbor distances), `d_max`, pop the most distant neighbor, and add `nc` to the k nearest neighbors.
    3. Otherwise, evaluate which child node should be evaluated first
        1. According to `nc`'s split dimension `s_d` and threshold `s_v`, find dimension `m_1` to evaluate
        2. Repeat from step 1 for `m_1` until return
        3. Evaluate if the other dimension,  `m_2` needs to be explored.
            1. Caculate the split distance of `m_2` on dimension `s_d` to the point, `d_split`
            2. if `d_split < d_max`, explore `m_2`, otherwise we don't.

The worst case scenario of KD Tree is to traverse through the entire tree. That is, `O(n Log(n))`, which could be worse than the brute force. But most cases, KD Tree is much faster. To speed up searching, we can moderately sacrifice precision and recall by utilizing "approximated nearest neighbor" (ANN):

- Approximated nearest neighbor is `d_split < alpha * d_max`
- KD Tree without approximated nearest neighbor should be able to find all K nearest neighbors

**[Implementation](https://github.com/RicoJia/Mumble-Robot/tree/main/mumble_onboard/halo)**

### NanoFLANN: what's a leaf node size?

In many textbook descriptions of kd trees, each leaf is thought of as holding a single point, and internal nodes only store splitting thresholds along a chosen dimension. However, in practical implementations such as nanoflann, the tree is built as a "bucket kd tree." Here's how it works:

1. The tree is built by recursively splitting the dataset. At each internal node, nanoflann chooses a **splitting dimension (typically the one with the largest spread)** and a splitting threshold (often the median of the points along that dimension). Points are partitioned into two groups: those with coordinates below (or equal to) the threshold go to the left child, and those above go to the right child.
2. Leaf Nodes as Buckets:
    - Instead of splitting until each leaf holds exactly one point, nanoflann **stops splitting when the number of points in a node is less than or equal to a specified maximum** (provided as a parameter in `KDTreeSingleIndexAdaptorParams`, e.g., 10). At that point, the node becomes a leaf node.
    - Leaf nodes in nanoflann do not represent a single point; they are buckets that contain several points. When a nearest neighbor query reaches a leaf node, it performs a linear search over all the points in that bucket.

## Octo Tree

Octo-Tree is similar to KD tree, it spatially segments a point cloud into tree nodes. It is a tree data structure in which each internal node has exactly eight children. Octrees are most often used to partition a three-dimensional space. In KD Tree, we have "splits", whereas in octotree, we have 3D boxes.

Tree building:

1. Input: point group
    1. Initialize the octo tree with 1 node, which is bounded by the min and max values of `x, y, z`
2. If the current point group is empty, return
3. If the current point group has 1 point, add a new leaf node to the tree and return.
4. If the current point group has more than 1 point:
    1. Create 8 children under the current node.
    2. Calculate the bounding boxes of each children. For each dimension in `x,y,z`, we create the bounding boxes by `0.5 * current_boundary`
    3. Partition the point group into 8 subgroups
    4. For each child, repeat from step 1 to insert their subgroup of points

Tree search:

1. Input: current tree node, query point p, K
2. Output: list of K nearest neighbors
3. For the current node `nc`:
    1. If `nc` is a leaf,
        1. calculate its distance to p, `d_cp`
        2. If `d_cp` < max(current_k_neighbor distances), `d_max`, pop the most distant neighbor, and add `nc` to the k nearest neighbors.
    3. Otherwise, evaluate which child node should be evaluated first
        1. Find boxes [`m_1`, ... `m_8`] that is closest to the query point in **ascending order**.
        2. For each box,
            1. Evaluate `d_box < d_max`. If not, return. If so, repeat from step 1 for `m_1` to update the k-nearest-neighbor list until return.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/eee970f8-1b64-41e0-ae68-f769f927781c" height="300" alt=""/>
       </figure>
    </p>
</div>

## Performance Summary

The speeds of above methods (Query Per Second, QPS) are:

```
grid search > KD tree >Octo Tree >> Brute Force
```

In reality, we 80% precision and recall is acceptable for point cloud registration.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/f05a9a66-cfc1-4595-bd69-5d1de961c87a" height="400" alt=""/>
       </figure>
    </p>
</div>
