---
layout: post
title: C++ Serial Communication
date: 2023-06-23 13:19
subtitle: select, read
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Linux
---

## Opening a Port

```cpp
fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
```

**Flags:**

- `O_RDWR`: open the device file for both reading and writing.
- `O_NOCTTY`: prevents the device from becoming the process's controlling terminal.
  - Without this flag, if the device sends a signal byte like `0x03` (Ctrl-C), it could deliver `SIGINT` to your program.
  - Especially important for background processes and daemons.
- `O_NONBLOCK`: opens the device without blocking. Normally, opening a serial port can block until the device is ready.

**DTR (Data Terminal Ready):** Many USB serial devices (e.g., Arduino, motor controllers) use DTR, a control signal in RS-232 serial communication from Data Terminal Equipment (DTE, such as a computer) to Data Communications Equipment (DCE).

## Port Configuration: `termios`

`termios` is a POSIX struct for configuring serial port behavior: baud rate, parity, stop bits, raw vs. canonical mode, echo, signals, etc.

## Reading from a Serial Port

Read `count` bytes from a file descriptor with a timeout:

```cpp
// Use select() to block at most `timeout_ms` waiting for data,
// rather than spinning or blocking indefinitely.
size_t received = 0;
while (received < count) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd_, &fds);

    struct timeval tv;
    tv.tv_sec  = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    int sel = ::select(fd_ + 1, &fds, nullptr, nullptr, &tv);
    if (sel <= 0) {
        return false;   // timeout (sel == 0) or error (sel < 0)
    }

    ssize_t n = ::read(fd_, buf + received, count - received);
    if (n <= 0) {
        return false;
    }
    received += static_cast<size_t>(n);
}
```

**Key functions:**

- `select()`: blocks until the file descriptor is ready or the timeout expires.
  - Return values: `sel > 0` — data ready; `sel == 0` — timeout; `sel < 0` — error.
  - No busy-waiting: the process sleeps in the kernel, freeing the CPU for other work.
  - Can monitor multiple file descriptors simultaneously (sockets, serial ports, pipes, stdin, etc.).
- `read(fd_, buf + received, count - received)`: reads up to `count - received` bytes into `buf` at offset `received`.
