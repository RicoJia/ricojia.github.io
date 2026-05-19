---
layout: post
title: ROS2 - Packages
date: 2025-1-11 13:19
subtitle: Pure Python Packages
header-img: img/post-bg-os-metro.jpg
tags:
  - Robotics
  - ROS2
comments: true
---
## ROSPy

for a **pure ROS 2 Python package**, you usually do **not** need a CMakeLists.txt. There are two valid ROS 2 Python package styles:

1.  ament_python style, No CMakeLists.txt.

```
package.xml
setup.py
setup.cfg
resource/<package_name>
<package_name>/__init__.py
```

In package.xml:

```
<buildtool_depend>ament_python</buildtool_depend>

<export>
  <build_type>ament_python</build_type>
</export>
```

2. **ament_cmake + ament_cmake_python style**

```
CMakeLists.txt
package.xml
<package_name>/__init__.py
```

This is what point_cloud_compressor does. It is still Python code, but the package is built through CMake. You need CMakeLists.txt for that style
