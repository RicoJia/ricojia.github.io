---
layout: post
title: OpenCV and Eigen Tools
date: '2024-05-09 13:19'
excerpt: A Running List of Opencv and Eigen Tools
comments: true
---

## OpenCV Image Tools

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

## Eigen