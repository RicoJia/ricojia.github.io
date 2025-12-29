---
layout: post
title: Robotics - EIGEN_MAKE_ALIGNED_OPERATOR_NEW
date: 2024-10-22 13:19
subtitle: Eigen
header-img: img/post-bg-miui6.jpg
tags:
  - Robotics
comments: true
---
### What does `EIGEN_MAKE_ALIGNED_OPERATOR_NEW` do exactly?

- Eigen **vectorizable** types (e.g. Vector4d, Quaterniond, small fixed‐size Matrix<…>) carry an alignment requirement (usually 16 bytes, sometimes 32 on AVX machines).

    ```cpp
    template<typename Scalar, int Rows, int Cols /*…*/>
    struct Matrix
    {
        // this expands (on compilers that support it) to:
        // alignas(EIGEN_MAX_ALIGN_BYTES)
        EIGEN_ALIGN16
        Scalar data[Rows * Cols];
    };
    using Vector4d = Matrix<double,4,1>;
    ```

  - In compile time, Vector4d expands to

 ```cpp
 struct alignas(16) Vector4d {
  double data[4];
  // …
 };
 ```

- The standard `::operator new` on some platforms (or older compilers) doesn’t promise alignments above what the C++ ABI requires (often only 8 or maybe 16 bytes).
  - In C++ (since C++11), there’s a special POD type called `std::max_align_t` (in `<cstddef>`) whose sole purpose is to have an alignment requirement at least as big as that of any scalar (fundamental) type on your platform
  - opeator new `void *operator new(std::size_t);` must return memory that is suitably aligned for any object whose alignment requirement does NOT exceed `alignof(std::max_align_t)`
  - On many platforms, `std::max_align_t = 16`

- Macro `EIGEN_MAKE_ALIGNED_OPERATOR_NEW` will yield memory that is actually aligned to EIGEN_MAX_ALIGN_BYTES (16, 32, etc).
  - But when using `operator new`, or STL containers (which calls operator new in  `T* allocate(std::size_t n) {}`), we need to use the Macro to override the global new / delete for your class. It injects below definitions of into your class, and makes each call go through Eigen’s aligned‐allocation routines (EIGEN_ALIGNED_MALLOC / EIGEN_ALIGNED_FREE).

```cpp
void* operator new(std::size_t size);
void  operator delete(void* ptr);
void* operator new[](std::size_t size);
void  operator delete[](void* ptr);
```

- **Therefore, EVERY TIME your class contains fixed-size Eigen, add EIGEN_MAKE_ALIGNED_OPERATOR_NEW to your class/struct**

    ```cpp
    struct Pose {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Sophus::SO3d    rot;    // internally holds a quaternion (4 doubles)
        Eigen::Vector3d trans;  // 3 doubles
    };
    ```

## Add `EIGEN_MAKE_ALIGNED_OPERATOR_NEW when you have all of the following

1. The class has Eigen members like:
1. `Eigen::Vector4d`, `Eigen::Matrix4d`, `Eigen::Quaterniond`
2. `Eigen::Isometry3d`, `Eigen::Affine3d`, `Eigen::Transform<...>`
3. any fixed-size type whose size is a multiple of the SIMD packet (common: 16 bytes for SSE, 32 for AVX)
2. The class can be allocated dynamically:
1. `new MyType`
2. `std::make_shared<MyType>`
3. stored in containers like `std::vector<MyType>` (or `std::vector<std::shared_ptr<MyType>>` if the objects themselves are heap-allocated)
4. or generally lives in places where alignment isn’t guaranteed by default allocators (older toolchains especially)

### When you usually don’t need it

1. The object is **only** stack-allocated (`PoseThing x;`) and you’re on modern compilers/platforms (stack alignment is typically OK).
2. The class only has **dynamic-size** Eigen types (`Eigen::VectorXd`, `MatrixXd`) — those manage their own aligned heap buffers internally.
3. You never store it in STL containers by value / never allocate it on heap.
