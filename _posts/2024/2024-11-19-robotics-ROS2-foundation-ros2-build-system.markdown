---
layout: post
title: Robotics - [ROS2 Foundation 3] ROS2 Build System
date: '2024-11-19 13:19'
subtitle: colcon-build, interface package, dpkg, rosdep
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

### ROS 2 Cpp And Python Package

[Reference](https://roboticsbackend.com/ros2-package-for-both-python-and-cpp-nodes/)

1. We’ll create a ROS2 Cpp package, which contains a package.xml and CMakeLists.txt.

    ```bash
    cd ~/ros2_ws/src/
    ros2 pkg create my_cpp_py_pkg --build-type ament_cmake
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

## Packaging & Build System

- In ROS2, cpp files require `CMakeLists.txt`, python files require `setup.cfg` and `setup.py`:

```
cpp_package_1/
    CMakeLists.txt
    include/cpp_package_1/
    package.xml
    src/

py_package_1/
    package.xml
    resource/py_package_1
    setup.cfg
    setup.py
    py_package_1/
```

- With Colcon, I like:

```
colcon build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=1 --packages-select dummy_test --cmake-force-configure
```

- `-DCMAKE_BUILD_TYPE=RelWithDebInfo`: This sets the CMake variable CMAKE_BUILD_TYPE to RelWithDebInfo, meaning “Release with Debug Info.”. Despite the existence of the debugging symbols, below can still happen with optimization:
  - Lines can be merged or removed
  - Variables can vanish
  - Stepping can feel “jumpy”
  - use a pure Debug build (no optimization) or something like -Og (for GCC) or -O1 -g (for Clang) for real debug build

- `-DCMAKE_EXPORT_COMPILE_COMMANDS=1`: Tells CMake to generate a compile_commands.json file in your build directory. This JSON file lists all compiler invocations for your project, which is extremely useful for tools like clangd, code analyzers, and IDEs that need to know your include paths and compiler flags.
- `--cmake-force-configure` This is not a standard CMake flag; it’s a colcon (ROS 2 build tool) argument. It forces CMake to re-run its configuration step for all packages, even if CMake thinks nothing has changed.

### `rosdep`

[rosdep page](https://docs.ros.org/en/humble/Tutorials/Intermediate/Rosdep.html)

`rosdep` will:

1. check for `package.xml` files in its path or for a specific package and find the rosdep keys stored within.
    - The dependencies in the package.xml file are generally referred to as “rosdep keys”.
    - Build tags
        - `<depend>` are dependencies that should be provided at both build time and run time for your package.
            - For C++ packages, if in doubt, use this tag.
            - Pure Python packages generally don’t have a build phase, so should never use this and should use `<exec_depend>` instead.
        - `<exec_depend>` declares dependencies for shared libraries, executables, Python modules, launch scripts and other files required when running your package.

2. Query keys in a central index to find the appropriate ROS packages
    - Retrieving the central index on to your local machine (`/etc/ros/rosdep/sources.list.d/20-default.list`) so that it doesn’t have to access the network every time it runs
3. Install the ROS packages

### Finding A Package Depedency and Version

If I have installed `behaviortree_cpp` in the `/opt` space, and I want to inspect its version,

1. `dpkg -l | grep behaviortree`

```
ii  ros-humble-behaviortree-cpp                        4.7.1-1jammy.20250513.175053            amd64        This package provides the Behavior Trees core library.
ii  ros-humble-behaviortree-cpp-v3                     3.8.7-1jammy.20250429.201614            amd64        This package provides the Behavior Trees core library.
```

2. `ros2 pkg behaviortree_cpp`: the path to ros-installed binary

```
└─  $ ros2 pkg prefix behaviortree_cpp
/root/my_ws/install/behaviortree_cpp

cd /root/my_ws/install/behaviortree_cpp/lib
dpkg -l | grep behaviortree_cpp
ii  ros-humble-behaviortree-cpp                        4.7.1-1jammy.20250513.175053            amd64        This package provides the Behavior Trees core library.
ii  ros-humble-behaviortree-cpp-v3                     3.8.7-1jammy.20250429.201614            amd64        This package provides the Behavior Trees core library.
```

- The binary is in the `/lib` directory
- To see `package.xml`, etc, go to `cd /root/my_ws/install/behaviortree_cpp/share`. This can be checked using `ros2 pkg prefix behaviortree_cpp --share`

## C++ & Python Interface Packages

An interface package defines ROS2 messages, services, and common utilities for other ROS2 packages. To create Python utilities, once a Python package is built and installed, and the workspace is sourced, its Python modules in `install/MY_INTERFACE/local/lib/python3.10/dist-packages/MY_INTERFACE` are automatically added to the Python path. What we need are as follow:

- Make sure the python package files are here:
  - Empty `MY_INTERFACE/MY_INTERFACE/__init__.py`
  - Module file: `MY_INTERFACE/MY_INTERFACE/My_Module`
- Add `MY_INTERFACE/setup.py`:

    ```python
    from setuptools import setup, find_packages

    package_name = "MY_PACKAGE"

    setup(
        name=package_name,
        version="0.0.1",
        packages=find_packages(include=[package_name, f"{package_name}.*"]),
        install_requires=["setuptools"],
        zip_safe=True,
        maintainer="TODO",
        maintainer_email="your_email@example.com",
        description="TODO",
        license="Apache-2.0",
        tests_require=["pytest"],
        entry_points={
            "console_scripts": [],
        },
    ) 
    ```

- `CMakeLists.txt`

    ```c
    install(
        DIRECTORY mumble_interfaces/
        DESTINATION local/lib/python3.10/dist-packages/mumble_interfaces
        FILES_MATCHING
    PATTERN "*.py")
    ament_package()
    ```

  - The destination `install/MY_INTERFACE/local/lib/python3.10/dist-packages/MY_INTERFACE` **is carefully chosen**, because that's where generated srv, msg files go. Why? Because when there are packages with the same name at two different locations, Python will look into one, and throw a `file-not-found` error if files are not there.
  - We are NOT using `ament_python_install_package` because it's meant for pure Python packages. We need to manually install `MY_INTERFACE` in `install/MY_INTERFACE/lib/python3.10/site-packages`

- User Code:

```python
from MY_INTERFACE import MyFunc
```

    - Or one can use: `python3 -c import MY_INTERFACE.My_Module` in the same console, because after sourcing `install/setup.bash`, the installed file is added to the Python Path. 
    - One can check the python path with: `python3 -c "import sys; print(sys.path)"` or `echo $PYTHONPATH`

## `colcon build`

### `--symlink-install`

- Without `--symlink-install`: copies artifacts into `install/`.
- With the flag: replaces many of those copies with symlinks for quicker iteration.

| Mode                    | What goes into `install/…`                                                                                                             | Practical consequences                                                                                                            |                                                                                                                                                |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Default (no flag)**   | A *copy* of every file produced by each package’s normal **install** step (executables, `.so` libraries, Python packages, resources…). | Safe and self-contained, but every rebuild recopies files; editing a Python script in `src/` has **no effect** until you rebuild. |                                                                                                                                                |
| **`--symlink-install`** | Wherever possible, **symbolic links** that point back into the build or source trees instead of copies.                                | Much faster iteration during development: change a Python file, re-source \`setup.\[bash                                          | zsh]`, and run again—no rebuild needed. Not all artifacts can be symlinked (e.g. versioned`.so\` chains that CMake creates are still copied). |

More specifically, without `--symlink-install`, these files are copied:

- Build tree of each package, e.g. `${WS}/build/my_pkg/lib/libmy.so ${WS}/install/my_pkg/lib/libmy.so` - Compiled binaries and shared objects produced by CMake/ament
- Source tree for Python packages, e.g. `${WS}/src/my_py_pkg/my_py_pkg/*.py` (after `setup.py` install runs inside the build step) `${WS}/install/my_py_pkg/lib/python3.x/site-packages/my_py_pkg/*.py` Python modules, entry-point scripts
- Resource files declared with install(DIRECTORY …) or ament_index_register_resource, e.g., Launch files, URDFs, icons, etc.

**Note, symlinks are NOT what `ldd` gives - **ldd resolves all shared library dependencies of an executable**, [while a symlink is a separate inode whose data is the path to a another inode](https://ricojia.github.io/2018/01/10/linux-filesystems/)**

### When to Rebuild?

Scenario 1: Sometimes, My ros2 binary didnt seem updated: when I updated a file in behavior_executor

- Solution1: `rm -rf build/<BINARY> install/<BINARY>`. Afterwards, --cmake-clean-cache starts working, without it is also fine
  - One caveat is, I could build in `~/MY_WS/src` (not in `~/MY_WS`). This means you might see a successful build, without actually building in the right place
- Solution2:  `--cmake-clean-first`
  - This flag tells colcon to run the CMake “clean” target before building each CMake‐based package. In practice:
    - Without `--cmake-clean-first`, colcon will do an incremental build: it only rebuilds targets whose inputs (source, CMakeLists, headers) have changed, and it won’t clear out old artifacts.
  - However, note: It **doesn’t purge the CMake cache**. If you need CMake to re-run the configure step (e.g. you changed `find_package` flags or `toolchain settings`), you’ll also want: `--cmake-force-configure`

Scenario 2: How to build a package from source:

- Clone it
- Build the package ONLY with `--symlink-install`
- Optional: Can mv the package binary you are trying to replace. E.g,

    ```
    sudo mv /opt/ros/humble/lib/libbehaviortree_cpp.so
    ```

- Source the setup.bash, then run it/ `colcon build --packages-select capacity_manager behavior_executor  behaviortree_cpp   --cmake-clean-first --allow-overriding behaviortree_cpp`
