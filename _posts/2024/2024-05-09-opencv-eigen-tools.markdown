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

- Create a cv::Mat

```cpp
//
cv::Mat(another_mat);
```
    - **Watch out**, if you pass another mat directly into `cv::Mat()` constructor, then the new mat will use the same pointer to the underlying data structure. To create a real copy, do `cv::Mat.clone()`

- Check image size:

```cpp
cv::Mat mat;
int rows = mat.rows();
int cols = mat.cols();
cv::Size s = mat.size();
rows = s.height;
cols = s.width;
```

- Find min and max values
```cpp
double min, max;
cv::Mat mat
cv::minMaxIdx(mat, &min, &max);  // Find min and max depth values
```

- `cv::Rect{}` is a commonly used data structure for storing upper left and bottom right coords. In OpenCV, the bottom right corner is **not** included, the top left corner is included. It can be instantiated mostly in two ways: 

```cpp
// Instantiating with top left corner and width, and height
auto rec1 = cv::Rect{1,2, 2,2};
std::cout<<"rec 1 br: "<<rec1.br()<<std::endl;  // see (3,4)
// Instantiating with top left corner and bottom right corner
auto rec2 = cv::Rect{cv::Point(1,2), cv::Point(3,4)};
std::cout<<"Rec 2 br: "<<rec2.br()<<std::endl;  // see (3,4)
// see 0, because row 3 and column 4 are not included
std::cout<<"contains: (3,3)"<<rec1.contains(cv::Point(3,3))<<std::endl;
```

## OpenCV Math Tools

- Basic Representation of points
    - `cv::Point2f` is 32 bit single precision floating point
    - `cv::Point2i` is 32-bit single precision integer.
- Rotation Vector and Matrix:
```cpp
cv::Mat r, R;
// cv::Mat_<double>(3,1) is a template function?
r = (cv::Mat_<double>(3,1) << 0, 0, CV_PI/2);
cv::Rodrigues(r, R);
```

## OpenCV Misc Tools

- `waitkey(timeout)` - wait for key press. If `timeout=0`, this will be infinite

    - This function needs an image widget active. Otherwise, it would just become a delay.


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
