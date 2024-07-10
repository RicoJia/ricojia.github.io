---
layout: post
title: CMake Thingamagigs
date: '2024-05-10 13:19'
excerpt: A Running List of CMake Thingamagigs
comments: true
---

## Concepts

Global 

## Advanced Options

- `ccache` is a compiler cache that caches previous compilations and detecting if the same compilation needs to be done again. It's meant for C/C++ projects.
    - How to use it

    ```cmake
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    endif()
    ```

    - Standalone: `ccache gcc -o myprogram myprogram.c`
    - [How it works](https://ccache.dev/manual/4.8.2.html#_how_ccache_works)
        1. Create a hash for each compilation using a fast cryptographic hash algorithm. Information for hashing include:
            - file info such as current dir, modification time
            - compiler name
            - input source file, compiler options (direct mode)
        2. Look up for existing hash, which includes **a cache hit or cache miss**
        
