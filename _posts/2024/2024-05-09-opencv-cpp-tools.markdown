---
layout: post
title: OpenCV Tools
date: '2024-05-09 13:19'
excerpt: A Running List of Opencv Tools
comments: true
---

## OpenCV Image Tools

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