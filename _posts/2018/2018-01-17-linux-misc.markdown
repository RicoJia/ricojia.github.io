---
layout: post
title: Linux Miscellaneous Thingies
date: '2018-01-17 13:19'
subtitle: Window Focus, Screenshots, File Differences, Formats, Shortkeys, UART, Tmux, ffmpeg
comments: true
tags:
    - Linux
---

## Screenshots

- `sudo apt install gnome-screenshot`
- Open `keyboard shortcuts -> add custom shortcuts`
- I use `ctrl-alt-4` to enable `gnome-screenshot -a`

## Window Focus

On my Ubuntu 22.04 system, when trying to download webpages / images from Chrome / Firefox, I noticed that the download dialog box is not focused in the first place. After searching for auto-focusing, I came up with a sub-optimal solution: **auto-shift to window that the mouse currently hovers on.**

1. `sudo apt install gnome-tweaks`
2. `Press the Super (Windows) key and type Tweaks, then press Enter.`
3. Choose `focus on hover`,
    - If you don't want to see the other window that's out of focus already, choose "Raise Window When Focused"
4. Press Alt + F2, type r, and press Enter. **This restarts the GNOME Shell without logging out.**

## File Differences

- `diff` shows which files are different and which files show in both places

```bash
diff -qr filter_projects/ Fun_Projects/filter_projects/
Only in filter_projects/: build_project.sh
Files filter_projects/face_tracker/face_tracker.cpp and Fun_Projects/filter_projects/face_tracker/face_tracker.cpp differ
Files filter_projects/face_tracker/face_tracker.hpp and Fun_Projects/filter_projects/face_tracker/face_tracker.hpp differ
```

- If in a git repo, see when the directory was last modified:

```bash
git log -1 --format="%ci" -- filter_projects/
2022-03-31 12:07:13 -0500
```

- `-1`: Limits the output to the most recent commit.
- `--format="%ci"`: Formats the output to display the commit date in ISO format.

## File Formats

- `AppImage`: a linux file format that's directly runnable without unzipping, debian instalation, etc. This might require `libfuse` to run: `sudo apt install libfuse`
    - `libfuse` is a library that uses `FUSE` (Filesystem in Userspace). We can create and manage file systems in the user space instead of the kernel space. 
        - It can create a virtual filesystem, like AppImage, where it's mounted as a virtual filesystem.
        - It works with `SSHFS`, mounting remote file systems via SSH

## System Commands

- `hostname`: host machine name in "USERNAME@HOSTNAME"

## Hot keys

- If to swap the function of `Fn` when hitting F2 and mute, `Fn lock` is the way to go. Just do `Esc + Fn`

## Hardware Interfaces

### `gpiochip` Interface

The `gpiochip` interface in Linux represents a GPIO (General Purpose Input/Output) controller. It allows users to interact with GPIO pins on embedded systems, SoCs, or microcontrollers.

- The active-high state means that a logical 1 corresponds to a high voltage level.
- Some GPIOs are configured as inputs, while others are outputs.
- `gpioinfo` to check such info
- `ttyACMx` is for serial-USB interface. `ttySx` is for native Hardware-Serial interface

```
/dev/ttyACM0  (Main USB Serial - SERIAL_DEBUG)
```

### Linux Device Tree

Linux Device Tree is a hardware description of the layout of hardware components in embedded linux systems. (Buses, peripherals, memory). With it, we don't need to modify the kernel source code.

- Device tree source (DTS) files define the config, and are compiled into **device tree blobs** (DTB), which the kernel load at boot.
- A device tree snippet:

```
pinctrl_uart3: uart3grp {
    fsl,pins = <
        MX8MP_IOMUXC_UART3_RXD__UART3_DCE_RX  0x40
        MX8MP_IOMUXC_UART3_TXD__UART3_DCE_TX  0x40
    >;
};
```

- Many embedded processors have **multiplexed I/O pins** that are muxed (multiplexed) to a specific functionailty, like I2C, SPI, UART, GPIO
    - `MX8MP_IOMUXC_UART3_RXD__UART3_DCE_RX` configures UART3
    - `MX8MP_IOMUXC_UART3_TXD__UART3_DCE_TX`: configures UART3 TX
    - `0x40` is a pad control register configuration for electrical characteristics like pull-up/down, drive strength, etc. 
    - UART2 and UART3 have the same pad control values. They don't necessarily cause issues as long as they are mapped to different pins.

- To verify serial being successfully configured at runtime:
    ```
    ls -l /dev/serial*
    dmesg | grep tty
    cat /proc/tty/drivers
    ```
    - check for `ttyUSBX` and `ttymxcX`
- To check pins at runtime: `cat /sys/kernel/debug/pinctrl/*/pins | grep UART`

`dmesg` (diagnostic message) prints **kernel ring buffer** that contains system logs for: boot process, hardware detection (UART, I2C, SPI, etc.), driver loading, kernel errors.

## UART Setup

- UART communication is typically a point-to-point protocol, meaning only one device should transmit (TX) while another listens (RX). However, UART is Push-Pull (actively driving line High/Low), not open-drain (like I2C), this could result in electrical conflicts.
    - Indeterminate bus values (data corruption)
    - High current draw (power dissipation due to short circuits?), and possible hardware damage

- Setting a processor pin to `GPIO` is safer, because it doesn't drive the bus. Instead it uses pull-up resistors or open-drain config.
    - When the GPIO pin is set to input, it has high impedance.


## Tmux

- `Ctrl + b, w`: all windows
- `Ctrl + b, n`: next window, `Ctrl + b, p`: previous window, `Ctrl + b, <number>`: numbered window
- `Ctrl + b, arrow`
- Horizontal split: `Ctrl + b, "`, vertical split: `Ctrl + b, %`
- scroll up: `c-b [`

```
tmux attach-session -t <WINDOW_NUM_OPTIONAL>
```

## Checksum

Most zip tools have checksum check built in, even though checksum is not part of the zip format

## ffmpeg

- cut a video from 8th second to 19th second: 
    ```
    ffmpeg -ss 8 -i ndt_sputnik2-2025-04-19_11.37.57.mp4 -to 19 -c copy cut_output2.mp4 
    ```
- Make the video 3x faster:
    ```
    ffmpeg -i cut_output2.mp4 -filter:v "setpts=0.33*PTS" -an cut_output2x.mp4    
    ```

## FireFox & Chrome & PDF Ops

- Firefox browser may have trouble actually saving PDFs. I've had cases where hitting the save button there does not actually save.
    - Chrome is safe
    - qoppa PDF viewer 2024 is actually great! No watermark, full PDF editor with signature, textbox, etc.
    - For merging pdfs `pdfunite file1.pdf file2.pdf merged.pdf`
