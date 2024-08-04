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
