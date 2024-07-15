---
layout: post
title: RGBD SLAM Bundle Adjustment
date: '2024-07-11 13:19'
excerpt: RGBD SLAM Backend Introduction
comments: true
---

## Problem Setup

Three types of bundle adjustment:

- Feature graph
    - Each vertex is a camera pose / 3D point (feature)
    - Each edge is an edge that connects the observed 3D point to its camera pose. It represents the 2D reprojection error.
- Pose graph
    - each graph is a camera pose only
- Factor graph


## G2O Introduction

G2O has a list of the optimizations it does. Here is short excerpt of the items that pertain to our rgbd slam problem:

- slam3d: 
    - Each vertex represents
        - a robot pose with 6dof.
        - 3D points (landmarks)
    - Each constraint represents
        -  the pose-pose constraint.
- SBA (Sparse Bundle Adjustment):
    - Each vertex represents:
        - camera intrinsics (optional?)
        - extrinsics
        - 3D points (landmarks)???
    - Each constraint include:
        - 2D projection (with known intrinsics?) onto image plane (2D coordinates)
        - monocular projection with parameters (with unknown intrinsics?)
        - stereo projection (3D points can be projected back onto the left and right cameras. The baseline between the cameras shhould be known)
        - scale constraint between extrinsics nodes? (Used in scenario where additional information about the relative scale / distance between multiple cameras is known)

Linear solvers include:
- PCG
- colamd
- CHOLMOD
- csparse
- dense
- eigen

## How G2O Works

- Sparse Optimizer

- Marginalization TODO


## G2O Set Up

For least-squares problem

A **vertex** is the sets of parameters to optimize. In the context of slam, a vertex is the camera pose, i.e., $se(3)$ parameters (6 parameters). **A constraint, or an edge** is a measurement that was seen at least two camera poses, so in the graph, an edge connects at least two nodes.
Node error function?

Then, an optimizer's job is to 
- Find gradient of the total cost function at their vertices (i.e., adding up all constraints)
- Apply Levenberg-Marquardt on parameters under optimization to find a local (hopefully global) minimum.

In SLAM, a vertex needs to satisfy $se(3)$ requirements. In `g2o/types/sba/types_six_dof_expmap.h` these vertices can be defined:
- `VertexSE3Expmap` represents robot poses in SE3 space, 
- `VertexSBAPointXYZ` represents a 3D point
- `EdgeProjectXYZ2UV` represents the projection of a 3D point onto the image plane

G2O is used in famous SLAM algorithms like ORB_SLAM. 

[Example](https://github.com/RainerKuemmerle/g2o/blob/master/g2o/examples/ba/ba_demo.cpp)

## References

[1] Fan Zheng's Farm: https://fzheng.me/2016/03/15/g2o-demo/
[2] g2o "what is in these directories": https://github.com/RainerKuemmerle/g2o/blob/master/g2o/what_is_in_these_directories.txt

