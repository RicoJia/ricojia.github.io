---
layout: post
title: C++ Profiling and Speed Ups
date: '2023-05-24 13:19'
subtitle: gprof, perf, CMake Release and Debug BoilerPlate, CMake Settings
comments: true
header-img: "img/post-bg-infinity.jpg"
tags:
    - C++
    - CMake
---

## Gprof

GNU gprof provides CPU execution times (not wall time, so sleep is not accounted for) of functions and their subfunctions in percentage. Gprof outputs data in a text-based format, which can be difficult to interpret. This is where gprof2dot comes in—it converts the profiling data into a visual call graph that makes it easier to understand function relationships and execution costs.

Gprof limitations: 
    - It measures [user time, but not kernel time](https://ricojia.github.io/2018/01/20/linux-operating-systems/)

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/9818a9fb-a77c-48c7-a110-4655d8b7f6be" height="300" alt=""/>
            <figcaption><a href="https://codeyarns.com/tech/2013-06-24-how-to-visualize-profiler-output-as-graph-using-gprof2dot.html#gsc.tab=0">Source</a></figcaption>
       </figure>
    </p>
</div>

[Youtube Reference](https://www.youtube.com/watch?v=zbTtVW64R_I)

1. Compile your program with `-pg`: `gcc -pg -o my_program my_program.c` or `g++ -pg -o my_program my_program.cpp`. In CMake:

```c
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -pg")
```

- `-pg` is specifcally for GNU gprof. It
    - Inserts instrumentation code into the executable to monitor function calls and execution times.
    - Generates a profiling report (gmon.out) when the program finishes running.
-  **Profiling does not work well with compiler optimizations (-O2 or -O3)**. The compiler may **inline or reorder functions**, making gprof reports unreliable.
- Therefore, Debug mode is recommended for `gprof`:

```cpp
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -pg -O2")
```

2. Install `gprof2dot`: `pip install gprof2dot`. If this doesn't work, try:

```bash
git clone https://github.com/jrfonseca/gprof2dot.git
```

    - One can use `python3 -m gprof2dot <ARGS>` if the module has been installed successfully

3. Install dot (part of Graphviz):

```bash
sudo apt update
sudo apt install graphviz
```

4. Run the program:

```
gprof -p -q my_program gmon.out > analysis.txt
```
- `-p`: Only prints flat profile (function time usage)
- `-q`: Only prints call graph
- `-b` to suppress verbose explanations:

5. Create a `profile.png` view of the graph:

```
python3 gprof2dot.py -s -w analysis.txt | dot -Tpng -o profile.png
```

- `-w` means "wrap text in bounding box"
- `-s` strips away args

6. To check the times of a subtree:
    1. Check the full function signature

        ```bash
        python3 gprof2dot.py  --list-functions='*add_scan*'   -w ~/file_exchange_port/Mumble-Robot/mumble_onboard/profile.txt
        ```
    2. Generate the sub-tree:

        ```bash
        python3 gprof2dot.py -s --root='halo::IncrementalNDTLO::add_scan(std::shared_ptr<pcl::PointCloud<pcl::PointXYZI> >, bool)' -w ~/file_exchange_port/Mumble-Robot/mumble_onboard/profile.txt | dot -Tpng -o profile.png
        ```

## perf

- [Reference post](https://www.cnblogs.com/wx170119/p/11459995.html)

1. Compile your binary with `-g` to get debug symbols. Debug symbols could increase the size of your binary, but wouldn't cause slowdowns
2. Second, run

```bash
perf record -F 99 -a -g -- MY_EXE && perf script -i perf.data &> perf.unfold                        
```

3. Visualization:
    1. FlameGraph: `git clone https://github.com/brendangregg/FlameGraph.git`
        ```
        ./FlameGraph/stackcollapse-perf.pl perf.unfold &> perf.folded              
        ./FlameGraph/flamegraph.pl perf.folded > perf.svg
        ```
    2. [Speedscope (my favorite)](https://www.speedscope.app/)
        - Just drag `perf.unfold` onto the page!

### Perf in a Docker Container

Running perf inside Docker has a few quirks, especially with custom kernels like those from System76 or Pop!_OS. Here's a refined setup guide:

1. If you see this error:

```
WARNING: perf not found for kernel 6.9.3-76060903

  You may need to install the following packages for this specific kernel:
    linux-tools-6.9.3-76060903-generic
```

💡 Why: 

- Some systems (e.g., System76) use custom kernel builds that don’t have matching `linux-tools-<version>` packages in the default Ubuntu repos.
- Installing linux-tools-$(uname -r) inside the container won’t work if the package doesn’t exist in apt.

✅ Solution:

Perf's ABI has been very stable. It is backward and forward ABI compatible via `perf_event_open(2)`. You can use a perf binary from a different kernel version as long as the syscalls remain stable (which they do). Just make sure the binary matches your host architecture and brings any needed .so dependencies. 

Use the perf binary from the host, where it was successfully installed:

```
cp /usr/lib/linux-tools-$(uname -r)/perf ./perf-copy
```

- ⚠️ Do not copy /usr/bin/perf — it's just a symlink that may not point to the real binary.

2. Required docker-compose settings in `docker-compose.yaml`:

```yaml
privileged: true      # or at least cap_add: [SYS_ADMIN, PERFMON]
pid: "host"
```

- `privileged` or `SYS_ADMIN/PERFMON` are needed for `perf_event_open`
- `pid: "host"` is critical so `perf` can trace real PIDs and resolve symbols correctly.

3. If you see this warning:

```
Perf tool from other version of kernel still can be used (the syscalls in perf_event subsystem have good design and are compatible with older/newer tools). So, you can just find any perf binary (not the /usr/bin/perf script) anywhere, check its library depends with (ldd ..path_to_perf/perf) and copy perf inside Docker (and install libs).
```

Do this in the container:

```bash
sudo sysctl -w kernel.perf_event_paranoid=0
sudo sysctl --system
```

### Optick

Optick failed to run on my linux machine. I could not get JSON file out, only the `opt` file out. Also, it needs a GUI to run it. The gui only be built on MSVC.


## CMake Boiler Plate For Release and Debug

For a ROS 2 workspace, we generally want a structured CMakeLists.txt design that allows easy toggling between Debug and Release modes. Why? I worked on a KD tree implementation. The same implementation takes 0.18s for KD tree building with 20k 3D LiDAR Points under the Debug mode. Under the release mode? 3ms (60x speed up)!

In a two level structure `my_ros2_workspace -> halo`, We control optimization flags, profiling tools (gprof, gdb), and CPU-specific instructions from the **top-level workspace** CMakeLists.txt, while allowing package-specific settings in halo/CMakeLists.txt.

- Workspace-Level (`my_ros2_workspace/CMakeLists.txt`) (top-level) sets the default build type and allows toggling between Debug and Release modes dynamically.

```cpp
cmake_minimum_required(VERSION 3.10)
project(my_ros2_workspace)

# Ensure that a build type is set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug, Release, RelWithDebInfo)" FORCE)
endif()

# Define common flags for different build types
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Building in Debug mode with gprof and gdb support")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg -ggdb -O0")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -pg")
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
    message(STATUS "Building in Release mode with optimizations")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -O3")
    if(WITH_SSE)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -msse4.2")
    endif()
    if(WITH_AVX)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx2")
    endif()
endif()

# Export build type setting to subdirectories
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}})

# Add your packages
add_subdirectory(halo)
```

- Sub-Package Level `halo/CMakeLists.txt`: Each ROS 2 package (like halo) should inherit the build settings from the top-level `CMakeLists.txt`

```c
cmake_minimum_required(VERSION 3.10)
project(halo)
# Enable C++17 (or higher)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure build type is consistent
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug, Release, RelWithDebInfo)" FORCE)
endif()
# Use the common flags from the workspace-level CMakeLists.txt
add_definitions(${CMAKE_CXX_FLAGS})

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)
...
```

Some explanations:

- `CACHE`: Stores the variable persistently across CMake runs.
- `STRING`: Specifies that this is a string-type variable.
- `"Build type (Debug, Release, RelWithDebInfo)"`: A user-friendly message to describe the variable.
- `FORCE`: Overwrites any previously set value of CMAKE_BUILD_TYPE in the cache. This ensures that CMAKE_BUILD_TYPE is set globally and is not overridden by user settings unless explicitly changed.

Understanding the `WITH_SSE` and `WITH_AVX` Flags:

- These flags enable optional CPU-specific optimizations using SSE4.2 and AVX2 instruction sets. They are useful for performance-critical applications like computer vision, deep learning, or scientific computing.

CMake **automatically sets certain compiler flags depending on the build type**. The default values are:

- Debug usually includes `-O0` and `-g`.
- Release usually includes `-O3` (for GCC/Clang) and often `-DNDEBUG`.
- `RelWithDebInfo` usually includes -O2 plus debug symbols.
- `MinSizeRel` usually includes -Os (optimize for size).

If you only commented out your custom optimization lines, you did not override CMake’s built-in defaults for Release. By default, Release mode is still optimized (most often -O3).

To build with `colcon`:

- `colcon build --cmake-args -DCMAKE_BUILD_TYPE=Debug`
- `colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release`
- Enable WITH_SSE or WITH_AVX for Release: `colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DWITH_SSE=ON -DWITH_AVX=ON`

- `colcon_build_source --executor-args -j4`: `--executor-args` is used for things like number of threads `-j4`

To build with `CMake`:

- `cmake .. -DCMAKE_BUILD_TYPE=Debug`
- `cmake .. -DCMAKE_BUILD_TYPE=Release` (This applies -O3 for maximum optimization.)
- `cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo` (This enables -O2 optimizations while keeping debugging symbols.)

### Quirk: Seeing Segfault Or Inconsistent Values In Release Runs From The Debug Mode

This is almost certainly an undefined behavior. In Debug mode, **compilers often zero-initialize more aggressively or add padding/checks, so you get a “lucky” consistent result (like pointers)**. In Release mode, the uninitialized pointers can contain garbage values. So:

- Always initialize pointers (and all fields) in your structs/classes. Debug mode can mask uninitialized usage. Release mode typically reveals these bugs.

- If a function has a return type but is not returning anything, in release mode, we see **seg faults**

## Optional CMake Settings

- `set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)`: useful if we are building static lib: `.a` files. set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

## UNRECOMMENDED CMake Settings

- `set(CMAKE_CXX_FLAGS "-w")` - suppress all warnings
- `set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g -ggdb ${CMAKE_CXX_FLAGS}")` This preserves debugging symbols while applying `O2` optimizations. But instead, one can just use `colcon build --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo`
