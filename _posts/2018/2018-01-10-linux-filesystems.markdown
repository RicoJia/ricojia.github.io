---
layout: post
title: Linux - Filesystem
date: 2018-01-10 13:19
subtitle: inode, soft/symbolic link, FIFO, File Lock
comments: true
tags:
  - Linux
---

## Inode

On every Unix-style filesystem (ext4, XFS, APFS, etc.), each regular file, directory, FIFO, … is represented on disk by a tiny metadata record called an inode (“index node”). That's like a file's identity card

| Stored in the inode                                 | Not stored in the inode           |
| --------------------------------------------------- | --------------------------------- |
| *File type* (regular, dir, symlink, …)              | The file’s **name**               |
| *Permissions & mode* (`rwxr-xr-x`, set-uid bits, …) | The full pathname                 |
| *Owner & group IDs*                                 | Which directory the file lives in |
| *Timestamps* (ctime, mtime, atime)                  | -                                 |
| *Size* & block/extent pointers                      | -                                 |
| *Link count* (how many hard links point here)       | -                                 |

Inside the filesystem, a directory is just **a table** with rows (or **directory entry**) like

```
<FILENAME>   →   <inode number>
```

- `FILENAME` is something like `vacation.jpg`
- The **inode number** is an integer key that the **kernel** uses to find the inode structure on disk; that **inode holds the object’s metadata and the list of disk blocks** where its data live.
  - The metadata inode has includes: type, size, timestamps, block pointers, …

This way you can have many hard links (filenames TODO?) all referring to the same directory inode. Also, this means renaming a file does not change the file's data, only the directory data?

## Soft and Hard Links

### Soft Link

Symlinks are a.k.a softlinks. They are a separate inode whose data is a **text path** to the target. Deleting or moving the target leaves the link dangling. It can be:

- The pathname can point to a file on another mounted volume.
- It's easily breakable, if the target's filepath has changed.
- `ls -l` shows `lrwxrwxrwx 1 … -> target`
- Create a symlink: `ln -s /path/tofile linkname`

### Hard Link

Because any directory entry can point to any inode on the same filesystem, you can create more than one entry that references the exact same inode number. Such an entry is a "hardlink"

```
ln original.txt copy.txt      # hard-link
```

After that:

```
$ ls -li
12345 -rw-r--r-- 2 alice users … original.txt
12345 -rw-r--r-- 2 alice users … copy.txt
```

Both names map to inode 12345; the link-count field (the third column) is now 2. These two names are equal peers: delete either one (rm original.txt) and the file’s data stay on disk

Some notes are:

- Hard-linking a directory is disallowed on almost all Unix systems to prevent infinite directory loops; the only directory hard links that exist are the automatic “.” and “..” entries that every directory contains.
- It does not work across file systems.
- On most linux systems, they cannot point to directories.
- If the target's filepath has changed, this wouldn't be broken
- `ls -l` shows `identical size/perm as original file`

### Example of cross-filesystem boundary

Imagine your laptop has:

- A primary root filesystem mounted at `/`
- An external USB drive auto-mounted at `/media/usbdrive`

You keep your photo library on the USB drive but want **a convenient alias in your home directory**:

```bash
ln -s /media/usbdrive/Photos/2024/Vacation ~/Pictures/Vacation2024
```

**Note that hard links cannot be created cross file-system boundaries**

## FIFO (a.k.a Named Pipe For Its Behavior)

A **FIFO special file** is a rendez-vous point in the filesystem that **lets two or more processes exchange a byte stream**, **first-in/first-out**, just like the anonymous pipe you get from pipe().

- You create one with `mkfifo mypipe`.
- The directory entry **holds no data**; the kernel moves the bytes directly between processes once both ends are open.
- Because it lives in the directory tree, unrelated processes can open it by name (unlike an anonymous pipe).

## Copying

- `rsync -vva --info=progress2 <FROM> <TO>`

## File Lock

Linux file locks are a simple way to coordinate access to the same file across multiple processes. They are commonly used to prevent conflicts such as two copies of the same program running at once, or multiple processes trying to update the same resource simultaneously. File locks are mainly a **process-level coordination** mechanism, not a guarantee enforced against every possible file access pattern.

A common way to do this in Python is with `fcntl.flock()`. For example:

```bash
import fcntl

try:
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    print("Lock acquired")
except BlockingIOError:
    print("Another process already holds the lock")
```

In this example, `LOCK_EX` requests an **exclusive lock**, meaning only one process can hold it at a time, while `LOCK_NB` makes the request **non-blocking**. If another process already holds the lock, the call fails immediately instead of waiting.

One important detail is that Linux file locks are generally **advisory**. This means other processes must also cooperate by using the same locking mechanism. A process that ignores locking can still access the file normally. Because of this, file locks work best when all participating programs are designed to respect them.

Another subtle point is that the lock applies to the **open file description**, not simply the filename itself. In practice, this means the lock is tied to the opened file handle and its descriptor, not just to the path on disk.

The lock is released in any of these situations:

- when you explicitly call `fcntl.flock(fd, fcntl.LOCK_UN)`
- when the file descriptor is closed
- when the process exits
