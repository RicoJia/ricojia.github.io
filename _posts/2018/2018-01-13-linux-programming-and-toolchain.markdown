---
layout: post
title: Linux - Programming And Toolchain
date: '2018-01-13 13:19'
subtitle: ldd
comments: true
tags:
    - Linux
---

## ELF and Dynamic Linking

Executable and Linkable Format (ELF) is the standard binary file format on Linux, most BSDs, and many embedded systems. It replaces older a.out and COFF formats.

| Aspect                     | Summary                                                                                                                                                                                                                                                                                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**                | Encapsulate **everything** the OS or linker needs to load an **executable, shared library (`.so`), object file (`.o`), core dump, or kernel module**.                                                                                                                                                                                                         |
| **Key parts (simplified)** | *ELF Header* → tells the loader “I’m 64-bit little-endian, entry point is 0x401000, program-header table starts at …”<br>*Program Header Table* → runtime view: segments to map (text, data, dynamic info).<br>*Section Header Table* (absent in stripped binaries) → link-time view: named sections like `.text`, `.rodata`, `.dynsym`, `.debug_info`. |
| **Dynamic linking**        | One segment (`PT_DYNAMIC`) lists required shared objects and relocation info; at runtime the dynamic loader (`ld.so`) walks this table to satisfy `libm.so.6`, `libstdc++.so.6`, etc. Tools such as **`ldd`** simply read that table and show you what the loader will do.                                                                              |
| **Extensibility**          | New segment or section types can be added without breaking older loaders; that’s how features like ASLR hints, GNU RELRO, and CET notes slid in over the years.                                                                                                                                                                                         |
| **Contrast**               | Windows uses **PE/COFF**; macOS uses **Mach-O**. The roles are analogous but the layout and metadata differ.                                                                                                                                                                                                                                            |
ELF is both the link-time and runtime container, it covers static executables, position-independent shared-libraries, tiny bootloaders, and even kernel images.

- `ldd program` - reads ELF's dynamic section to list library dependencies
- `readelf -a file` or `objdump -p file` - dumps headers symbol tables, relocation entries

### ldd - Inspect Shared-Library Dependencies

`ldd some_executable` asks the dynamic linker to resolve each shared object name (they are loaded during runtime), and prints `libsomething.so.1 => /actual/path/libsomething.so.1 (0xaddress)`

```bash
└─  $ ldd ./lib/libbehavior_executor.so
 linux-vdso.so.1 (0x0000728663d27000)
 libdiagnostic_msgs__rosidl_typesupport_cpp.so => /opt/ros/humble/lib/libdiagnostic_msgs__rosidl_typesupport_cpp.so (0x0000728663aaa000)
```
