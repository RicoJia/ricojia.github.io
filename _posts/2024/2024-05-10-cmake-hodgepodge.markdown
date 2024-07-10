---
layout: post
title: CMake Thingamagigs
date: '2024-05-10 13:19'
excerpt: A Running List of CMake Thingamagigs
comments: true
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
        
