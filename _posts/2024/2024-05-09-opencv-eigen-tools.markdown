---
layout: post
title: OpenCV and Eigen Tools
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

#### Check Image Size

```cpp
cv::Mat mat;
int rows = mat.rows();
int cols = mat.cols();
cv::Size s = mat.size();
std::cout<<"image size: "<<img.size<<std::endl;
rows = s.height;
cols = s.width;
```

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
