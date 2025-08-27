---
layout: post
title: Computation Tools - OpenCV, Eigen, Sophus, PCL, PCD, NanoFLANN
date: '2024-05-09 13:19'
subtitle: A Running List of Opencv and Eigen Tools
comments: true
tags:
    - OpenCV
---

## OpenCV Image Processing

### cv::Mat

#### Create a cv::Mat

In C++:

```cpp
// method 1
cv::Mat(another_mat);
// method 2
cv::Mat A = (cv::Mat_<uchar>(1,2)<<1,2);    //CV_8U
// method 3
cv::Mat image = cv::Mat::zeros(height, width, CV_8UC1);
// method 4: fixed size matrix
cv::Matx31d point(k.pt.x, k.pt.y, 1.0);
```

- **Watch out**, if you pass another mat directly into `cv::Mat()` constructor, then the new mat will use the same pointer to the underlying data structure. To create a real copy, do `cv::Mat.clone()`

To copy an image, in python:

```python
output_img = cv2.cvtColor(input_img.copy(), cv2.COLOR_GRAY2RGB)
```

#### Check Image Size and Indexing

```cpp
cv::Mat mat;
int rows = mat.rows();
int cols = mat.cols();
cv::Size s = mat.size();
std::cout<<"image size: "<<img.size<<std::endl;
rows = s.height;
cols = s.width;
```

- Opencv indexing has `row` coming before `col`: `image.at<cv::Vec3b>(row, col)`

#### Find min and max values

```cpp
double min, max;
cv::Mat mat
cv::minMaxIdx(mat, &min, &max);  // Find min and max depth values
```

### cv::Rect

`cv::Rect{}` is a commonly used data structure for storing upper left and bottom right coords. In OpenCV, the bottom right corner is **not** included, the top left corner is included. It can be instantiated mostly in two ways:

```cpp
// Instantiating with top left corner and width, and height
auto rec1 = cv::Rect{1,2, 2,2};
std::cout<<"rec 1 br: "<<rec1.br()<<std::endl;  // see (3,4)
// Instantiating with top left corner and bottom right corner
// y represents rows, x represents columns
auto rec = cv::Rect{cv::Point(0,0), cv::Point(1,2)};
std::cout<<"Rec 2 br: "<<rec.br()<<"width: "<<rec.width<<std::endl;  // see (1,2), width is 1
// see 0, because row 1 and column 2 are not included
std::cout<<"contains: (1,2)"<<rec1.contains(cv::Point(1,2))<<std::endl;
```

- How to set a block of matrix to a certain value in `cv::Mat`

```cpp
constexpr const int radius = 10;
cv::Mat image = cv::Mat::zeros((radius+1) * 2, (radius+1) * 2, CV_8UC1);
cv::Rect second_quadrant(0, 0, radius+1, radius+1);
image(second_quadrant) = 1;
```

- How to shift a `cv::Rect`:

```cpp
// integers
cv::Point shift_point(5, 10);  // Example shift values
cv::Rect rect(0, 0, 100, 50);  // Example rectangle

// float points
cv::Point2f shift_point(5.5f, 10.5f);  // Example shift values
cv::Rect rect(0, 0, 100, 50);          // Example rectangle

// Convert the rectangle's top-left to Point2f and shift. 
rect + cv::Point(shift_point);
cv::Point(shift_point) + rect;  // this does not hold
```

## Random Access

- `cv::MatExpr` is an intermediate data type that does not have `at()`. So for random access, one needs to:

```cpp
cv::Matx31d point(k.pt.x, k.pt.y, 1.0);
// BAD
auto rotated_pt = rotation_matrix * point;
// GOOD
cv::Mat rotated_pt = rotation_matrix * point;
return {rotated_pt.at<int>(0), rotated_pt.at<int>(1)};
```

## OpenCV Math Tools

- Basic Representation of points
  - `cv::Point2f` is 32 bit single precision floating point
  - `cv::Point2i` is 32-bit single precision integer.
  - **Note**: `+` is NOT supported for `cv::Point` types

- matrix operations:
  - dot product: `a * b`
  - element-wise product: `cv::multiply(mat1, mat2, result);`
- Rotation Vector and Matrix:

```cpp
cv::Mat r, R;
// cv::Mat_<double>(3,1) is a template function?
r = (cv::Mat_<double>(3,1) << 0, 0, CV_PI/2);
cv::Rodrigues(r, R);
```

- Append another matrix to an existing one

```cpp
if (!mat1.empty()){
    cv::vconcat(mat1, mat2, result); // vertically append mat2 to mat 1 and store it in result
    cv::hconcat(mat1, mat2, result); // horizontally append mat2 to mat 1 and store it in result
}
else{
    result = mat2;
}
```

    - **IMPORTANT: `cv::vconcat(src, nsrc, dst)` requires `src` to be not empty!!** 

- Calculate `atan2`: `cv::fastAtan2(y, x)`. Its accuracy is about 0.3 deg

### Integral Image

- [`cv::integral`](https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga97b87bec26908237e8ba0f6e96d23e28), which sums up all pixels from the top left corner to the given pixel. The result integral image will have a size `[column+1, row+1]`, and the value at (y,x) represents the sum from [0,0] to [y-1, x-1]. Left and top borders of the integral image are padded with 0.

$$
sum(X,Y) = \sum_{x<X, y<Y} I(x,y)
$$

    - `cv::integral()` supports only `cv_8U`, `CV_32S` and `cv_32U`. So use `img.convertTo(img, cv_8U)` if necessary

## OpenCV Misc Tools

- `waitkey(timeout)` - wait for key press. If `timeout=0`, this will be infinite

  - This function needs an image widget active. Otherwise, it would just become a delay.

### OpenCV Errors

- ```Unsupported combination of source format (=4), and buffer format (=5) in function 'getLinearRowFilter'``` means the input type needs to be changed (e.g., int -> float)

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

## Attention! Quirks of Eigen

### Lazy Evaluation Could Cause Issues in Eigen (Version 3.4.0)

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

### Eigen Matrix Does NOT support dynamic matrix appending Efficiently

If we want to dynamically add rows to a matrix, unfortunately, I think it'd be more efficient to have a `vector<vector<double>>`, then convert it into a matrix all at once. This is because Eigen assumes that we know the size of the matrix beforehand. So otherwise, we need to resize the matrix constantly. One quirk of eigen::matrix::resize is that it resets values as well.

```cpp
Eigen::MatrixXd mat(2, 2);
mat << 1, 2,
        3, 4;
std::cout << "Before resize:\n" << mat << "\n\n";
mat.resize(2, 2);  // resize to the same size!
```

### What does `EIGEN_MAKE_ALIGNED_OPERATOR_NEW` do exactly?

- Eigen vectorizable types (e.g. Vector4d, Quaterniond, small fixed‐size Matrix<…>) carry an alignment requirement (usually 16 bytes, sometimes 32 on AVX machines).

    ```cpp
    template<typename Scalar, int Rows, int Cols /*…*/>
    struct Matrix
    {
        // this expands (on compilers that support it) to:
        //    alignas(EIGEN_MAX_ALIGN_BYTES)
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

## PCD

`pcd` stands for Point Cloud Data, and it’s the standard file format used by the `Point Cloud Library (PCL)`. It stores 3D points (and `optionally color`, `normals`, `intensity`, etc.).

## NanoFLANN

- We can trust the distance values returned by the kd tree - it represents `dist(neighbor, point)`
