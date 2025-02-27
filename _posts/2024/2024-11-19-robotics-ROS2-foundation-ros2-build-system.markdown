---
layout: post
title: Robotics - [ROS2 Foundation 3] ROS2 Build System
date: '2024-11-19 13:19'
subtitle: Various Things to Note For Building Dockerized ROS2 App
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
    - Docker
comments: true
---

## Build Tool 

A build tool operates on **a set of packages**
1. determines the dependency graph 
2. invokes the specific build system for each package in topological order.
3. for a specific package, knows how to setup the environment for it, invokes the build, and sets up the environment to use the built package. 

The build system operates on a single package: `CMake`, `Make`, `Python setuptools`. `catkin` and `ament_cmake` are based on `CMake`

### Dependency Graph

- `find_package` helps the graph. `FindFoo.cmake` or `FindFoo.cmake` for the dependency must be in a prefix that CMake searches implicitly, like `/usr`, or a location provided through env vars `CMAKE_PREFIX_PATH`, or `CMAKE_MODULE_PATHCMAKE_MODULE_PATH`
- Install a shared_lib in a non-default location, that location needs to be in `LD_LIBRARY_PATH`.

## ROS 2 Cpp And Python Package

[Reference](https://roboticsbackend.com/ros2-package-for-both-python-and-cpp-nodes/)

1. We’ll create a ROS2 Cpp package, which contains a package.xml and CMakeLists.txt. 

    ```bash
    $ cd ~/ros2_ws/src/
    $ ros2 pkg create my_cpp_py_pkg --build-type ament_cmake
    ```
    - See:

        ```
        my_cpp_py_pkg/
        # --> package info, configuration, and compilation
        ├── CMakeLists.txt
        ├── package.xml
        # --> Python stuff
        ├── my_cpp_py_pkg
        │   ├── __init__.py
        │   └── module_to_import.py
        ├── scripts
        │   └── py_node.py
        # --> Cpp stuff
        ├── include
        │   └── my_cpp_py_pkg
        │       └── cpp_header.hpp
        └── src
            └── cpp_node.cpp
        ```

2. For Python, no more setup.py and setup.cfg, everything will be done in the CMakeLists.txt.
    - Note that we have a sub directory called `"my_cpp_py_pkg"`. Inside, the python code lives there. Like ROS1, they are executables `chmod +x`.
    - The module is visible to other packages: `from my_cpp_py_pkg.module_to_import import ...`

3. `package.xml`:

    ```xml
    <?xml version="1.0"?>
    <?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
    <package format="3">
    <name>my_cpp_py_pkg</name>
    <version>0.0.0</version>
    <description>TODO: Package description</description>
    <maintainer email="your@email.com">Name</maintainer>
    <license>TODO: License declaration</license>

    <buildtool_depend>ament_cmake</buildtool_depend>
    <buildtool_depend>ament_cmake_python</buildtool_depend>

    <depend>rclcpp</depend>
    <depend>rclpy</depend>

    <test_depend>ament_lint_auto</test_depend>
    <test_depend>ament_lint_common</test_depend>

    <export>
        <build_type>ament_cmake</build_type>
    </export>
    </package>
    ```

4. `CMakeLists.txt`:

    ```C
    cmake_minimum_required(VERSION 3.5)
    project(my_cpp_py_pkg)
    # Default to C++14
    if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
    endif()
    if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    endif()
    # Find dependencies
    find_package(ament_cmake REQUIRED)
    find_package(ament_cmake_python REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(rclpy REQUIRED)
    # Include Cpp "include" directory
    include_directories(include)
    # Create Cpp executable
    add_executable(cpp_executable src/cpp_node.cpp)
    ament_target_dependencies(cpp_executable rclcpp)
    # Install Cpp executables
    install(TARGETS
    cpp_executable
    DESTINATION lib/${PROJECT_NAME}
    )
    # Install Python modules
    ament_python_install_package(${PROJECT_NAME})
    # Install Python executables
    install(PROGRAMS
    scripts/py_node.py
    DESTINATION lib/${PROJECT_NAME}
    )
    ament_package()
    ```