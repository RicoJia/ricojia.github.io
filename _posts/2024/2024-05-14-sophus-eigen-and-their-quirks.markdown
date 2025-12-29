---
layout: post
title: Robotics - Eigen, Sophus And Their Quirks
date: 2024-10-22 13:19
subtitle: Sophus
header-img: img/post-bg-miui6.jpg
tags:
  - Robotics
comments: true
---

## Eigen Tools

### Common Conversions

- ROS `geometry_msgs::Pose` $->$  `Eigen::Quaterniond`

```cpp
geometry_msgs::Pose p;
Eigen::Quaterniond q(pose.rotation());
```

### Tricks

- Print a vector on one line

```cpp
// Define the IO format to print on one line
Eigen::IOFormat eigen_1_line_fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " [", "] ");

// Print the vector using the defined format
std::cout << "p1: " << p1.format(eigen_1_line_fmt) << std::endl;
```

---

## Attention! Eigen Quirks

### [Quirk 1] Writable Vectors Can Be Written Directly As A Copy

```cpp
for (auto r : normalized_mat.rowwise()){
    r -= medians;
}
```

- [colwise() or rowwise() are writable vector, +=, -= are valid operations](https://libeigen.gitlab.io/eigen/docs-nightly/classEigen_1_1DenseBase.html#a6daa3a3156ca0e0722bf78638e1c7f28)

### [Quirk 2] Lazy Evaluation Could Cause Issues in Eigen (Version 3.4.0)

In Eigen, expressions like:

```cpp
getter(item) - mean
```

may not be immediately evaluated due to lazy evaluation. `getter(item)-mean` is an `Eigen::CwiseBinaryOp` expression object that just stores references to `sum and getter(item)`. The subtraction might not execute as expected, potentially leading to unexpected behavior—such as returning a zero vector instead of the intended result.

The fix is:

```cpp
(getter(item)- mean).eval() 
```

Even this **does not** work:

```cpp
mean = std::accumulate(
    data.begin(), data.end(),
    VectorType::Zero().eval(),            // init value, a *concrete vector*
    [&getter](const VectorType &sum, const auto &item) {
        return sum + getter(item);        // <-- returns a *lazy expression*
    }
).eval() / static_cast<double>(N);
```

[Reference](https://github.com/gaoxiang12/slam_in_autonomous_driving/pull/191)

### Check Version

```cpp
std::cout << "Eigen version: " 
            << EIGEN_WORLD_VERSION << "." 
            << EIGEN_MAJOR_VERSION << "." 
            << EIGEN_MINOR_VERSION << std::endl;
```

### [Quirk 3] Eigen Matrix Does NOT support dynamic matrix appending Efficiently

If we want to dynamically add rows to a matrix, unfortunately, I think it'd be more efficient to have a `vector<vector<double>>`, then convert it into a matrix all at once. This is because Eigen assumes that we know the size of the matrix beforehand. So otherwise, we need to resize the matrix constantly. One quirk of eigen::matrix::resize is that it resets values as well.

```cpp
Eigen::MatrixXd mat(2, 2);
mat << 1, 2,
        3, 4;
std::cout << "Before resize:\n" << mat << "\n\n";
mat.resize(2, 2);  // resize to the same size!
```

### [Quirk 5] Do NOT use `std::accumulate`, use `+=`

```cpp
template <typename Container, typename VectorType, typename Getter>
void compute_cov_and_mean(const Container &data, VectorType &mean,
                          VectorType &cov, Getter &&getter) {
    ... 

    cov = std::accumulate(data.begin(), data.end(), VectorType::Zero().eval(),
                          [&mean, &getter](const VectorType &sum,
                                           const auto &item) {
                              auto diff = (getter(item).eval() - mean);
                              return (sum + diff.cwiseProduct(diff)).eval();
                          }) /
          static_cast<double>(N - 1);
}
```

In this snippet, it was observed that eigen vectors in `std::accumulate` could produce garbage values. `std::accumulate` creates a temporary vector using the object's copy constructor. One assumption I'm making here is Eigen vector's copy constructor may not be super robust. So, write this piece using `+=` instead

---

## Sophus

In Sophus, the SE(2) group represents a rigid 2D transformation, which consists of both a rotation and a translation. The operation you're looking for, SE(2) * Eigen::Vector2d, is a common need when transforming 2D points between coordinate frames.

```cpp
using Vec2d = Eigen::Vector2d;
using SE2 = Sophus::SE2d;
// Example: SE(2) transformation
SE2 transform(Eigen::Rotation2Dd(M_PI / 4), Vec2d(1.0, 2.0)); // 45-degree rotation and translation (1,2)
Vec2d point(3.0, 4.0);
Vec2d transformed_point = transform * point;
```

- Sophus SO2 uses a unit complex number: `[cos(θ), sin(θ)]` as its internal representation instead of a single value. This is because:

```
z = cos(θ) + i·sin(θ)
z₁ · z₂ = cos(θ₁ + θ₂) + i·sin(θ₁ + θ₂)
```

- This can avoid numerical instability near +-pi.

To construct a rotation: we need to use `Eigen::Quaterniond`

```cpp
ground_truth_pose.so3() = halo::SO3(Eigen::Quaterniond(qw, qx, qy, qz));
```

Altogether with translation:

```cpp
ground_truth_pose = halo::SE3(Eigen::Quaterniond(qw, qx, qy, qz), halo::Vec3d(x, y, z));
```

- Sophus SE3 stores an Eigen Quaternion (4 scalars, 32B) and a vector3d (3 scalars, 24B). Because they are fixed-size obj, they are packed into 16 byte boundaries. So in total, an `sophus::SE3` is 64B.

```cpp
class SE3{

    SO3Member so3_; // has QuaternionMember unit_quaternion_; where QuaternionMember is Eigen::Quaternion<Scalar, Options>
    TranslationMember translation_; // Vector3<Scalar, Options>
};
```

### Sophus Quirks

It was observed that sophus SE3 multiplications and inverse are not very stable. On some machines, there were garbage values after multiplying.. **in Release mode, there's uninitialized memory being used somewhere that's causing incorrect computation**.
So when I use it, I would provide below overrides

```cpp
struct SO3 : public Sophus::SO3d {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Base = Sophus::SO3d;
    using Base::Base;   // bring in all Sophus::SO3d ctors

    // Template overload for other Eigen vector expressions (e.g., Vec3d::Zero(), subtraction results)
    template <typename Derived>
    typename std::enable_if<
        std::is_same<typename Derived::Scalar, double>::value &&
            Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 1,
        Eigen::Vector3d>::type
    operator*(const Eigen::MatrixBase<Derived> &v) const {
        return this->matrix() * v;
    }

    // Override SO3 * SO3 to return halo::SO3
    SO3 operator*(const SO3 &other) const {
        return SO3(Base::operator*(other));
    }

    // SO3 * scalar (e.g., R * 2.0)
    template <typename Scalar, typename std::enable_if<std::is_arithmetic<Scalar>::value, int>::type = 0>
    Eigen::Matrix3d operator*(Scalar s) const {
        return this->matrix() * static_cast<double>(s);
    }

    // SO3 * Matrix (for Mat3d and other 3×N matrices)
    template <typename Derived>
    typename std::enable_if<
        std::is_same<typename Derived::Scalar, double>::value &&
            Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime != 1,
        Eigen::Matrix<double, 3, Derived::ColsAtCompileTime>>::type
    operator*(const Eigen::MatrixBase<Derived> &M) const {
        return this->matrix() * M;
    }

    // scalar * SO3 (e.g., 2.0 * R) - friend function for symmetry
    template <typename Scalar>
    friend typename std::enable_if<std::is_arithmetic<Scalar>::value, Eigen::Matrix3d>::type
    operator*(Scalar s, const SO3 &R) {
        return static_cast<double>(s) * R.matrix();
    }

    // Matrix * SO3 (e.g., Mat3d * R) - friend function
    template <typename Derived>
    friend typename std::enable_if<
        std::is_same<typename Derived::Scalar, double>::value &&
            Derived::RowsAtCompileTime == 3,
        Eigen::Matrix<double, Derived::RowsAtCompileTime, 3>>::type
    operator*(const Eigen::MatrixBase<Derived> &M, const SO3 &R) {
        return M * R.matrix();
    }
};

struct SE3 : public Sophus::SE3d {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Base = Sophus::SE3d;
    using Base::Base;   // <--- bring in all Sophus::SE3d ctors (Matrix4d, SO3+Vec3, etc.)

    using Base::operator*;   // keep all other Sophus::SE3d operator* overloads

    // Override SE3 * SE3 to return halo::SE3
    SE3 operator*(const SE3 &other) const {
        return SE3(this->so3() * other.so3(),
                   this->so3() * other.translation() + this->translation());
    }

    // Override SE3 * Vec3d for consistent behavior
    Vec3d operator*(const Vec3d &p) const {
        return this->so3().matrix() * p + this->translation();
    }

    // Override inverse to return halo::SE3 instead of Sophus::SE3d
    SE3 inverse() const {
        SO3 invR = this->so3().inverse();
        return SE3(invR, invR.matrix() * (this->translation() * -1.0));
    }
};
```
