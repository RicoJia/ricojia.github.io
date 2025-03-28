---
layout: post
title: Robotics - [Epic Bugs] SLAM Related Bugs
date: '2025-2-1 13:19'
subtitle: G2O
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - SLAM
comments: true
---

Below is a list of bugs that took me multiple hours, if not days, to troubleshoot and analyze. These are the "epic bugs" that are worth remembering for my career. 

## Epic Bug 1: G2O Optimization Didn't Update Vertex

### Summary

This bug took several hours (if not days) to debug. It appeared as if G2O was ignoring the optimization — the vertex (pose) remained unchanged despite running `optimizer.optimize(1)`

After ruling out common culprits like:

- Vertex not being added correctly
- Edges not referencing the right vertex
- Incorrect Jacobian implementation
- G2O configuration/setup issues

I eventually traced the issue to a subtle but critical misunderstanding in the error term formulation.

### Context 

In a point-to-line 2D ICP formulation, the error term e is typically calculated as the distance from a point to a line. The simplified (but effective) version of that is:

$$
\begin{gather*}
\begin{aligned}
& e = ap_x + bp_y + c
\end{aligned}
\end{gather*}
$$


Where a, b, and c define a line (ax + by + c = 0), and $(p_x, p_y)$ is the point.

In my case, the point came from source_cloud, expressed in the body frame. However, the line coefficients a, b, c were fit in the map frame, using nearest neighbors from the target cloud.

The Mistake

I precomputed the error term and stored it inside a struct:

```cpp
_error[0] = point_line_data_vec_ptr_->at.error_;
```

And upstream, this was assigned as:

```cpp
ret.at(idx).error_ = source_pt * point_coord; // Wrong!
```

The problem? This `source_pt` is in the body frame, and using it in the error term implies that optimization is being done relative to the body frame, not the map/submap frame. Because the error is now invariant to pose changes, optimization has no gradient — G2O doesn't change the pose, even if the edges are correctly wired.

### What Threw Me Off

- Point-to-line distances are frame-invariant.
- But scaled error terms like ap_x + bp_y + c are not.
- That mistake causes the optimizer to think the current pose is already optimal — so it just stays put.

It was like optimizing with the body frame assumed to be the map frame — **a silent bug with no crash or warning, just no progress.**

### The Fix

Don't precompute error_ using body frame coordinates. Instead, compute e = ax + by + c dynamically in computeError() using the transformed map-frame point.

The corrected version is now:

```cpp
class EdgeICP2D_PT2Line : public g2o::BaseUnaryEdge<1, double, VertexSE2> {
    .... 
    void computeError() override {
        auto *pose = static_cast<const VertexSE2 *>(_vertices[0]);
        double r   = source_scan_objs_ptr_->at(point_idx_).range;
        double a   = point_line_data_vec_ptr_->at(point_idx_).params_[0];
        double b   = point_line_data_vec_ptr_->at(point_idx_).params_[1];
        double c   = point_line_data_vec_ptr_->at(point_idx_).params_[2];
        double theta = source_scan_objs_ptr_->at(point_idx_).angle;
        Vec2d pw = pose->estimate() * Vec2d{r * cos(theta), r * sin(theta)};
        _error[0] = a * pw.x() + b * pw.y() + c;
    }
}
```

[Reference to the fixed code](https://github.com/RicoJia/Mumble-Robot/blob/cda3c4b9a14fbbd8e2bbe729f3b25ad01e3dfc48/mumble_onboard/halo/include/halo/2d_g2o_icp_methods.hpp#L26)


## Lessons Learned

- Optimized pipelines are hard to debug. To maximize vectorization, it's tempting to parallely calculate and store intermediate results. **However, if something goes wrong downstream, especially when we have a conceptual math error,it may stem from a silent assumption upstream.** 
- Coordinate frames matter: Even when the math looks simple, subtle frame mismatches can render your optimizer useless.
- Scaled point-to-line errors are not frame invariant: If you're using `ap_x + bp_y + c`, you must express the point in the same frame as the line.
- Verbose mode helps: G2O’s `setVerbose(true)` didn't show errors, but the chi² staying constant was a hint that nothing was being optimized.