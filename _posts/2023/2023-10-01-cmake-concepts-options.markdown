---
layout: post
title: CMake - Concepts and Options
date: '2023-10-01 13:19'
subtitle: CMake Concepts, Compile Options, Commands, CMake-Format, Header-Only Library, Static Library
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
    - Build Systems
---

## Concepts

CMake is a "write-only" language 😉, because it was only meant to be written, not for reading, jokingly. This is because:

- The syntax is not quite a real scripting language
- CMake uses a lot of macros (`target_include_directories, PUBLIC, INTERFACE`).
- Debugging is hard - error messages are bad; 
- Scoping rules are not always intuitive:

    ```c
    set (MY_VAR "global") 
    function (my_func)
        set(MY_VAR "local")
        message("Inside func: ${MY_VAR}")
    endfunction()
    my_func()
    message("Outside ${MY_VAR}")
    ```
    - In this example, we see "local", which is a actually a **new local variable** created
    - This is unlike Bash or Python where global variable are by default accessed
    - To access the parent scope:
        ```c
        function(my_func)
            set(MY_VAR "updated globally" PARENT_SCOPE)
        endfunction()
        ```

- Major version upgrades 
    - Pre CMake 3.x to CMake 3.x - make commands "target" specific:
        ```
        include_directories -> target_include_directories
        link_directories(${SOME_LIB_DIR}) -> target_link_directories
        ```
    - CMake: 3.13:
        ```
        target_link_options(my_app PRIVATE "-Wl,--no-as-needed")
        ```

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

### Find Package

```
find_package(PACKAGE REQUIRED)
```

- Requires `.cmake` files. If you have a static library, do not use find_package.

### Interface Library

Interface library is a library that does not create build artifacts, like `.a`, `.o` files. So you can use it to make a group of dependencies other targets can link against. Example:

```cmake
add_library(my_project_dependencies INTERFACE)
target_link_libraries(my_project_dependencies INTERFACE
    ${catkin_LIBRARIES}
    ${rosbag_LIBRARIES}
    simple_robotics_cpp_utils
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
)
add_executable(orb_test src/orb_test.cpp)
target_link_libraries(orb_test my_project_dependencies)
```

- Linking order: if `simple_robotics_cpp_utils` depends on `OpenCV`, it MUST come before it. Otherwise, there would be an `undefined reference` error to `OpenCV`

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

## Commands

- Read environment variable "CUSTOM_INSTALL_PATH": `$ENV{CUSTOM_INSTALL_PATH}/include`
- Setting a variable by concatenating two values `set(VAR var1 var2)`
  - `set(CMAKE_PREFIX_PATH "$ENV{CUSTOM_INSTALL_PATH}" ${CMAKE_PREFIX_PATH})`

### Advanced Options

- `ccache` is a compiler cache that caches previous compilations and detecting if the same compilation needs to be done again. It's meant for C/C++ projects.
    - How it works: 
        1. Caching: Ccache stores the output of compiler runs (object files) in a cache
        2. Ccache analyzes compilation parameters and source code content to determine if a cached version exists
        3. When you compile code that has already been compiled **with the same flags and source code**, ccache can retrieve the cached output instead of running the compiler again
    - Why it's useful:
        - ccache can help in situations where build systems might not be able to detect identical recompilations, such as when file timestamps are not reliable or when different workspaces are involved,
  - How to use it
        - ccache supports C, C++, Objective-C, and Objective-C++. 
    - Standalone: `ccache gcc -o myprogram myprogram.c`
    - Cmake:

            ```cmake
            find_program(CCACHE_PROGRAM ccache)
            if(CCACHE_PROGRAM)
                set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
                set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
                message(STATUS "CCACHE found, using ccache")
            else()
                message(STATUS "CCACHE not found, not using ccache")
            endif()
            ```

      - `RULE_LAUNCH_COMPILE` is a CMAKE property that specifies a command to be run before every compile.

  - [How it works](https://ccache.dev/manual/4.8.2.html#_how_ccache_works)
        1. Create a hash for each compilation using a fast cryptographic hash algorithm. Information for hashing include:
            - file info such as current dir, modification time
            - compiler name
            - input source file, compiler options (direct mode)
        2. Look up for existing hash, which includes **a cache hit or cache miss**

## Namespace

In CMake, a “namespace” is a way to qualify target names, helping to avoid collisions and clarify ownership—especially when your project or its dependencies might use generic names. For example, in `MyLib`, to create core:

```c
add_library(MyLib_core SHARED ${MyLib_CORE_SOURCES})
# ... other MyLib targets

# When installing/exporting, assign a namespace:
set_target_properties(MyLib_core PROPERTIES
    EXPORT_NAME core
)
# Then export them under the MyLib namespace:
install(TARGETS MyLib_core ...)
install(EXPORT MyLibTargets
    NAMESPACE MyLib::
    FILE MyLibTargets.cmake
    DESTINATION lib/cmake/MyLib
)
```

Then, in a consuming project, you would link to them as:

```c
target_link_libraries(MyApp PRIVATE MyLib::core MyLib::stuff ...)
```


## Header-Only Library vs Static Library 

### Header-Only Library

A header-only library in C++ is essentially a collection of .h (or .hpp) files that contain inline function definitions, templates, and constants without requiring a separate compilation step. This means that all the implementation details are in the headers, and there are no compiled object files (.o or .a) to link against.

**Drawback of Header-only Library**: Since header-only libraries do not produce a compiled object or shared library, the dependencies (e.g., PCL, g2o, Eigen, etc.) **must be explicitly included in each project that uses the library.**

In CMake, a header-only library is typically defined as an INTERFACE library using:

```cmake
add_library(my_header_lib INTERFACE)
target_include_directories(my_header_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
```

- **The INTERFACE keyword means that no compilation happens for this library**, and it only serves as an "interface" for consumers.
- When using this header-only library in another project, dependencies must still be explicitly added to target_link_libraries:

```cmake
target_link_libraries(my_project PRIVATE my_header_lib PCL g2o)
```

### Static Library

In a static library, all the **necessary object files (.o)** from the source files are compiled and packed into the .a file. When you link against the static library, these compiled objects are copied directly into the final executable. This means that all functions and symbols defined in the static library are available at runtime without requiring separate shared library files (.so or .dll).

To compile all necessary dependencies in the library, we need to: 

```c
target_link_libraries(my_static_lib PRIVATE PCL g2o)
```

- **PRIVATE ensures that PCL and g2o are only needed while compiling my_static_lib.a, but the user doesn't need to manually link them.** Now, when the user links my_static_lib.a, it should contain everything needed.

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

## `CMake-Format`

- Install `pip install cmake-format`
- Configure [`~/.cmake-format.py`](https://github.com/RicoJia/dot_files/blob/main/cmake-format.py)
- Use: `cmake-format -i CMakeLists.txt`
