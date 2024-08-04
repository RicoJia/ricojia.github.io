---
layout: post
title: CMake - Concepts and Options
date: '2023-10-01 13:19'
subtitle: A Running List of CMake Thingamagigs
comments: true
tags:
    - C++
---

## Concepts

### Hierarchy

CMake is hierarchical. That is, you have a top-level project, with multiple subprojects.

Basic Operations:

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyProject)

add_subdirectory(subproject1)
```

### Properties

- Global Properties

A global property is accessible throughout the entire project. **It's truly global, so once set in a sub project, it could be used in other subprojects.**
You can set a global property for compiler options, build configs, etc. 
Basic Operations:

```cmake
# top-level project CMakeLists.txt
set_property(GLOBAL PROPERTY <property_name> <property_value>)
# Retrieving the property in sub-level CMakeLists.txt
get_property(var GLOBAL PROPERTY <property_name>)
```

- Parent Scope Properties

In a sub-project, it has a reference to the parent it was launched in. So a property can be shared with parent project. [Here is a blog that further explains this](https://www.mgaudet.ca/technical/2017/8/31/some-notes-on-cmake-variables-and-scopes)

```cmake
# Sets FOO in the parent project, not the current scope
set(FOO <foo value> PARENT_SCOPE)
```

### Interface Library

Interface library is a library that does not create build artifacts, like `.a`, `.o` files. So you can use it to make a group of dependencies other targets can link against. Example:

```cmake
add_library(my_project_dependencies INTERFACE)
target_link_libraries(my_project_dependencies INTERFACE
    ${catkin_LIBRARIES}
    ${rosbag_LIBRARIES}
    ${OpenCV_LIBRARIES}
    simple_robotics_cpp_utils
    Eigen3::Eigen
)
add_executable(orb_test src/orb_test.cpp)
target_link_libraries(orb_test my_project_dependencies)
```

## Compile Options

To add more compile options, one simply does `SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} <OTHER OPTIONS>")`. Below is a list of options that can be used

- `-wall`: enable all commonly used warning messages during compilation.

### Choosing a good C++ Standard - what if the rest of the system does not support your standard?

- If the rest of the system does not support your chosen C++ standard, be aware of potential issues.
It is generally possible to use different C++ standards within the same project. However, be cautious of inconsistent ABI (Application Binary Interface) issues, particularly if the package is meant to be a library.
- If the package consists of standalone C++ modules, such as ROS nodes, mixing standards is usually acceptable as long as the modules do not share ABI-sensitive interfaces.

### Debug VS Release

`set(CMAKE_BUILD_TYPE Debug)` will include debug information so gdb can step through it, check variable information. It's equivalent to `gcc -g ...` using the gcc compiler.

- There wouldn't be much code optimziation in the debug mode. So the compilation time could be a bit shorter, but execution time would be longer.
- Debug symbols include: 
    - variable names
    - function names and params
    - line numbers
    - data types

`set(CMAKE_BUILD_TYPE Release)` not only exclude debug symbols, but will also turn on optimization, and disable assertions. This would include `-O3`

## Advanced Options

- `ccache` is a compiler cache that caches previous compilations and detecting if the same compilation needs to be done again. It's meant for C/C++ projects.
    - How to use it
        - Standalone: `ccache gcc -o myprogram myprogram.c`
        - Cmake:
            ```cmake
            find_program(CCACHE_PROGRAM ccache)
            if(CCACHE_PROGRAM)
                set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
            endif()
            ```
            - `RULE_LAUNCH_COMPILE` is a CMAKE property that specifies a command to be run before every compile.

    - [How it works](https://ccache.dev/manual/4.8.2.html#_how_ccache_works)
        1. Create a hash for each compilation using a fast cryptographic hash algorithm. Information for hashing include:
            - file info such as current dir, modification time
            - compiler name
            - input source file, compiler options (direct mode)
        2. Look up for existing hash, which includes **a cache hit or cache miss**
        
## Weird Issues 

- When `/usr/include/opencv4/opencv2/core/eigen.hpp:259:29: error: expected ‘,’ or ‘...’ before ‘<’ token 259 |Eigen::Matrix<_Tp, 1, Eigen::Dynamic>& dst )`. Solution:
    - **In header file, make sure eigen includes come before <opencv2/core/eigen.hpp>**
    
    ```cpp
    #include <Eigen/Core>
    #include <Eigen/Dense>
    #include <opencv2/core/eigen.hpp>
    ```
    
    - Make sure the CMake has:

    ```cmake
    target_include_directories(${PROJECT_NAME}
    PUBLIC 
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
    )
    
    target_link_libraries(executable 
        ...
        Eigen3::Eigen
    )
    ```