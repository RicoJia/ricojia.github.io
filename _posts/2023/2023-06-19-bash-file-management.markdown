---
layout: post
title: Bash File Management
date: 2023-06-19 13:19
subtitle:
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Linux
---
## File Management

`umask 077` #  permissions mask that ensures any new files you create in that shell session are readable and writable only by you, and not accessible to group members or others.

### A Faster Way To Copy Files From One Machine To Another Through SSH

```bash
ssh gpc1 'tar -C /tmp -cf - MY_DIR' | tar -C . -xf -
```

- This Command walks through the entire directory, and creates a tar stream that writes to stdout. SSH will execute this command, and carry that stream  to the local `tar` extraction.
- `tar` only reads files in the directory and writes to stream, so it's not like `scp` which also deals with timestamp, checksum, file skipping logic.
  - `rsync` is fast, but it still:
    - builds a file list on both sides
    - compares files: size, mime, perms, etc.
    - may do checksum logic (depending on flags)
    - lots of small protocol messages.
  - So if you want a fresh copy, rsync's incremental file delta method doesn't do much.
