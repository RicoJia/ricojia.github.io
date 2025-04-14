---
layout: post
title: Computation Tools - Eigen Expression Templates and SIMD
date: '2024-05-11 13:19'
subtitle: Fused Computation
comments: true
tags:
    - Eigen
---

## Eigen Expression Template. Optimization Impact: ⭐️⭐⭐️️⚪⚪

Expression templates are one of those “powerful but mysterious” C++ features that libraries like Eigen, Blaze, and even TensorFlow (C++) use to get zero-overhead performance while still writing math like normal algebra.


```cpp
struct Vec{
    Vec(int size): d_(std::vector<double>(size, 0.0)){}
    double& operator[](int i){return d_.at(i);}
    int size() const {return d_.size();}
private:
    std::vector<double> d_;

};

template<typename VecType>
struct Scaled {
    const VecType& v;
    double scale;
    // Not doing anything here
    Scaled(const VecType& vec, double s) : v(vec), scale(s) {}

    double operator[](int i) const {
        return scale * v[i];
    }
    int size() const { return v.size(); }
};

// Expression Template for adding two vectors
template<typename LHS, typename RHS>
struct Add {
    const LHS& lhs;
    const RHS& rhs;
    Add(const LHS& a, const RHS& b) : lhs(a), rhs(b) {}
    // Only evaluate when provided with an i
    double operator[](int i) const {
        return lhs[i] + rhs[i];
    }
    int size() const { return lhs.size(); }
};

// Overload * for scaling
template<typename VecType>
Scaled<VecType> operator*(double scalar, const VecType& v) {
    // So  we are not evaluating here
    return Scaled<VecType>(v, scalar);
}

// Overload + for addition
template<typename LHS, typename RHS>
Add<LHS, RHS> operator+(const LHS& lhs, const RHS& rhs) {
    // Not evaluating here
    return Add<LHS, RHS>(lhs, rhs);
}

template<typename Expr>
void assign(Vec& dest, const Expr& expr) {
    for (int i = 0; i < dest.size(); ++i)
        // Expr is either Add or Scaled
        // By calling expr[i], the overloaded [i] will be called in a chain. This minimizes tmp object creation
        dest[i] = expr[i];  // No temporaries, direct computation
}

int main() {
    Vec A(1000000), B(1000000), C(1000000);
    for (int i = 0; i < A.size(); ++i) {
        A[i] = i;
        B[i] = 2*i;
    }

    // Expression: C = A + 2 * B; — no temporaries created
    assign(C, A + 2 * B);

    std::cout << "C[100] = " << C[100] << std::endl;
}
```

As you can see:

- No evaluation is done when `+ or *` are called. The final evaluation happens when `exp[i]` is called in `assign()`. Then, all adds and multiplications are carried out, minimizing the creation of temporary objects.



### Expression Template's Issue with STL Functions

Expression templates are transformed into:

```cpp
Vec3d::Zero() -> Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, Eigen::Matrix<double, 3, 1>>
```

So if in an STL container, like in `std::accumulate`, it actually does

```cpp

```cpp
return std::accumulate(point_cloud->points.begin(), point_cloud->points.end(), Vec3d::Zero(),
        [](const Vec3d& sum, const PCLPointXYZI& point) {
            return sum + Vec3d(point.x, point.y, point.z);
        });
->
result = binary_op(result, *first);
```

Without `.eval()` will return another expression template. Then, when assigning back to `result`, it will cause an issue, because the previous expression template is not assignable.

So the rule of thumb here is, **always append `.eval()` inside STL functions**!

Another bug that's **extremely subtle** is that `sum + Vec3d(point.x, point.y, point.z)` returns an expression, ` (a CwiseBinaryOp object).`. So we need to force evaluation again. This can be done by:
1. Calling `eval()` explicitly
2. Forcing return type: `->Vec3d`

So the fixed version is:

```cpp
std::accumulate(point_cloud->points.begin(), point_cloud->points.end(), Vec3d::Zero().eval(),
            [](const Vec3d& sum, const PCLPointXYZI& point){
                Vec3d new_sum = sum + Vec3d(point.x, point.y, point.z);
                return new_sum;
            });
// Or
std::accumulate(point_cloud->points.begin(), point_cloud->points.end(), Vec3d::Zero().eval(),
            [](const Vec3d& sum, const PCLPointXYZI& point) -> Vec3d {
                return sum + Vec3d(point.x, point.y, point.z);
            });
// Or
std::accumulate(point_cloud->points.begin(), point_cloud->points.end(), Vec3d::Zero().eval(),
            [](const Vec3d& sum, const PCLPointXYZI& point){
                return sum + Vec3d(point.x, point.y, point.z).eval();
            });
```



## Why Expression Template? Fused Computation

Eigen makes extensive use of expression template to "delay" computation so different steps can be fused together to create a fused computation. For example, if we want to do `a +_2*b`, it's actually faster to do

```cpp
for (int i = 0; i < N; ++i){
    C[i] = A[i] + 2 * B[i];
}
```

Than:

```cpp
tmp = 2.0 * B;   // vector multiply
C = A + tmp;     // vector add
```

- This creates a `tmp`, which needs extra memory read/write for `tmp`.
- The loop is also easier for the optimizer to optimize

For maximum optimization, one can do:

```cpp
-O3 -march=native -funroll-loops -ftree-vectorize
```

This will actually expand to: `-march=haswell -msse4.2 -mavx2 -mfma -mbmi2 ...`

Don't trust me? [Check this code snippet and its assembly out, and run it yourself! 😃](https://godbolt.org/z/xbh6eabs4)

Some notes are:
- SIMD registers like `xmm0` can be found. But no SIMD instructions like `movapd` are used

