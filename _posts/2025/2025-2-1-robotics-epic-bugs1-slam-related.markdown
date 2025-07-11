---
layout: post
title: Robotics - [Bugs - 1] SLAM Related "Epic Bugs"
date: '2025-2-1 13:19'
subtitle: G2O Optimization Vertex Updates, Compiler-Specific Bugs, Yaml-cpp
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

## Epic Bug 2: Compiler Bugs

Here I'm not writing about "giant" bugs, but small tricky ones.

### Cannot Find Overloaded Operators

I had a bug where an `operator <<` is defined in `namespace1`. Because I spent most of my time developing within this namespace, I forgot that I should have included `namespace1` in its test, where namespaces are clearly indicated.

### Non-Dependent static_assert in `if constexpr` Always Fails In Older Compiler

In `gcc 14.2`, `if constexpr` can be evaluated properly. But in the snippet below, it cannot be evaluated properly in `gcc 10.1`. [Here is a proposal for the fix in new compiler](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2593r0.html)

[Here is a code snippet](https://godbolt.org/z/Pjfvqb6fn), but I'm posting here anyways

```cpp
#include <type_traits>
inline constexpr bool always_false = false;

template <typename T>
inline constexpr bool templated_always_false = false;

// this compiles in gcc 14.2
template <typename Foo>
void my_func() {
    if constexpr (std::is_same_v<Foo, int>) {
        // do something
    } else {
        // This line is fine because it is dependent on a template parameter, which forces evaluation in if constexpr?. 
        // So use it in older compilers
        static_assert(templated_always_false<Foo>, "Unsupported Foo type");

        static_assert(false, "Unsupported Foo type");
    }
}
```
- Use `gcc --version` to check your compiler's version!

### PCL Is A Worm Hole

- PCL does not support in-place filtering. Use a tmp cloud instead:
    ```cpp
    voxel_filter_.setInputCloud(local_map_);
    voxel_filter_.filter(*tmp_cloud_);
    // swap or copy the filtered result back into local_map_
    local_map_->swap(*tmp_cloud_);
    ```

### NanoFLANN

- When you create a temporary NanoflannPointCloudAdaptor and pass it by value into the KD-tree constructor, the adaptor object is destroyed as soon as the function exits. Since NanoFLANN only stores a raw pointer to your data, any subsequent query will dereference a dangling pointer—invoking undefined behavior on the very first lookup.

```cpp
nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>,
    PointCloudAdaptor,
    dim /* dimensionality */
>;
```

- NanoFLANN [tree search is thread safe](https://github.com/jlblancoc/nanoflann/issues/54)

## Epic Bug 3 - Eigen3("data is not aligned")

[Reference](https://github.com/RainerKuemmerle/g2o/issues/250)

[Chinese blog post]( https://blog.csdn.net/weixin_41353960/article/details/94738106) 
Setting `set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g ${CMAKE_CXX_FLAGS}")` actually wipes out CMake’s built-in “Release” flags (which are roughly -O3 -DNDEBUG) and replace them with exactly -O3 -g plus whatever happens to be in your global CMAKE_CXX_FLAGS

- CMake’s default release flags include -DNDEBUG, which tells both the C library’s assert() and Eigen’s alignment checks (via EIGEN_NO_DEBUG) to compile out all runtime assertions.
    - By omitting -DNDEBUG, all of Eigen’s mis-alignment asserts stay active → you hit “data is not aligned” at runtime.
- If your global CMAKE_CXX_FLAGS had -g, -O0, extra warning flags, etc., now they run in release too.
    - This both slows compile (duplicated flags to parse) and, worse, kills all optimizations if you didn’t explicitly keep -O3.

Fix: `set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g")`

## Epic Bug 4 - Yaml-cpp Node Assignment Actually Changes Underlying Object

Issue: If you copy-construct `new_node` from `root`, then assign it to another node, you actually change the what root points to:

``` cpp
[this, name, &slot](const YAML::Node &root) {
    YAML::Node new_node = root;  // creates a pointer to the underlying tree.
    new_node = root["seq"][0];   // This is the counter intuitive part - unlike pointer, this call actually changes the object root points to the subsction
    ...
};
```

This is because: 

- In [the copy constructor](https://github.com/jbeder/yaml-cpp/blob/2f86d13775d119edbb69af52e5f566fd65c6953b/include/yaml-cpp/node/impl.h#L45), we copy the pointer to memory, etc. 

```cpp
inline Node::Node(const Node&) = default;


inline Node::Node(const detail::iterator_value& rhs)
    : m_isValid(rhs.m_isValid),
      m_invalidKey(rhs.m_invalidKey),
      m_pMemory(rhs.m_pMemory),
      m_pNode(rhs.m_pNode) {}

```

- In [the assignment operator](https://github.com/jbeder/yaml-cpp/blob/2f86d13775d119edbb69af52e5f566fd65c6953b/include/yaml-cpp/node/impl.h#L206), we do `AssignNode`

```cpp
template <typename T>
inline Node& Node::operator=(const T& rhs) {
  Assign(rhs);
  return *this;
}
```

- Which, [in `AssignNode`](https://github.com/jbeder/yaml-cpp/blob/2f86d13775d119edbb69af52e5f566fd65c6953b/include/yaml-cpp/node/impl.h#L256), we are actually **UPDATE** the memory:

```cpp
inline void Node::AssignNode(const Node& rhs) {
  if (!m_isValid)
    throw InvalidNode(m_invalidKey);
  rhs.EnsureNodeExists();

  if (!m_pNode) {
    m_pNode = rhs.m_pNode;
    m_pMemory = rhs.m_pMemory;
    return;
  }

  m_pNode->set_ref(*rhs.m_pNode);
  m_pMemory->merge(*rhs.m_pMemory);
  m_pNode = rhs.m_pNode;
}
```