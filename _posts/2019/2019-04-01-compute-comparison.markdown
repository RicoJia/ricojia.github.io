---
layout: post
title: Compute Comparison
date: '2019-03-06 13:19'
subtitle: Arduino Uno vs Rpi Zero vs Rpi 4b+ vs Rpi 5 vs Nvidia Orin Nano Across Multiple Dimensions
comments: true
header-img: "img/post-bg-2015.jpg"
tags:
    - Python
---

## Memory

| Device       | RAM       | Cache (L1 / L2)       | Disk / Storage / Non-Volatile Memory       | Notes                                                                 |
|--------------|-----------|-----------------------|---------------------------------------------|-----------------------------------------------------------------------|
| Arduino UNO  | 2 KB      | ❌ None               | No Disk; Program is stored inFlash (32 KB); EEPROM (key-value)  | No File System                                                        |
| RPi Zero (ARM11, single-core)    | 512 MB    | 16 KB / 128 KB        | SD card                                     | Minimal Linux system, no virtual memory                                                  |
| RPi 4B (Cortex-A72, quad-core)      | 2–8 GB    | 32 KB / 1 MB (shared) | SD card / USB SSD                           | Full Linux computer-on-chip; Paging & Virtual Memory supported       |

