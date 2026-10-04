---
layout: post
title: "Linux - Storage: Filesystems, NVMe, SATA, and eMMC"
date: 2018-05-05 13:19
subtitle: What happens when a program reads a file, and how NVMe, SATA, and eMMC differ
comments: true
tags:
  - Linux
  - Embedded
---

## Filesystems

A **filesystem** is the system that organizes files and directories on a storage device. It keeps track of file names, where each file’s data lives, permissions, and available space. For how directories, inodes, and links work inside a filesystem, see [Linux - Filesystem](https://ricojia.github.io/2018/01/10/linux-filesystems/).

**ext4** means _fourth extended filesystem_. It’s a common Linux filesystem. Other examples include:

|Filesystem|Common use|
|---|---|
|**ext4**|Linux disks|
|**NTFS**|Windows disks|
|**APFS**|macOS disks|
|**exFAT**|USB drives and SD cards shared between operating systems|

To see which device and filesystem a directory lives on:

```bash
df -hT /storage/rico        # -> /dev/nvme0n1p1 ext4 1.8T, mounted on /storage
```

- `-h` prints human-readable sizes
- `-T` prints the filesystem type, such as `ext4`
- The device name hints at the storage type: `nvme0n1p1` is an NVMe drive, `mmcblk0p1` is eMMC or an SD card, and `sda1` is usually a SATA or USB drive.

## What Happens When a Program Reads a File

When your program reads a file, such as:

```python
data = np.load("/storage/rico/frame.npy")
```

the process is roughly:

1. **Linux checks its page cache in RAM:** if the data is already cached, no storage read is needed. See [Python - Numpy File Management](https://ricojia.github.io/2019/03/11/python-numpy-load-save/) for how this affects `np.save()` and `np.load()`.
2. **ext4 finds the data:** it translates positions in the file into **logical block addresses (LBAs)** on the device.
3. **The storage driver sends a read command:** “Read these blocks.”
4. **The device’s controller reads the flash** and transfers the data into RAM. The controller has its own mapping from logical blocks to physical flash pages, so Linux never sees where the data physically sits.
5. Your program receives the data.

Steps 3 and 4 depend on the storage interface: how the computer sends commands to the device, and how the device’s controller handles them.

## Storage Interfaces

NVMe SSDs, SATA SSDs, and eMMC all **store data in NAND flash memory**, which retains data without power. The difference is **how the computer sends commands** to them and **how their controllers handle** those commands.

A **hard disk drive (HDD) is the older alternative**. It stores bits as magnetic patches on **a spinning platter**, and [a head moves across the platter to reach them](https://www.youtube.com/watch?v=wteUW2sL7bc). Seeking is mechanical, so random reads are much slower than on flash.

### NVMe

NVMe (Non-Volatile Memory Express) is a protocol built specifically for flash SSDs. Its commands travel over **PCIe**, the computer’s high-speed expansion bus, which GPUs and network cards also use. NVMe drives are usually M.2 modules.

1. The computer puts commands into queues in RAM, then signals the SSD that work is ready.
2. The SSD processes those commands and uses **DMA (direct memory access) to transfer data into RAM**, without making the CPU copy every byte. It then reports completion.
3. NVMe supports up to 65,535 I/O queues with up to 65,536 commands each. That helps an SSD serve multiple programs and read different parts of a dataset in parallel.
4. Typical sequential speeds for an x4 drive are around **3,500 MB/s on PCIe 3.0**, **7,000 MB/s on PCIe 4.0**, and **10,000+ MB/s on PCIe 5.0**.

### SATA

SATA (Serial ATA) is an older standard originally designed for hard drives and later adapted for SSDs. Not every SSD is NVMe: a SATA SSD uses the same flash, but talks to the computer through the SATA interface and the AHCI protocol.

1. AHCI has a single command queue with up to 32 commands, which suited spinning disks but limits flash.
2. It uses a SATA cable/interface and typically tops out around **550 MB/s**, often in the familiar 2.5-inch drive form factor.

### eMMC

eMMC (embedded MultiMediaCard) packages a flash chip and its controller together, usually soldered onto the board. You find it on phones, low cost laptops, and embedded boards such as the Jetson Xavier.

1. The computer communicates with it through an **MMC bus**: a command line plus a parallel data bus of 1, 4, or 8 bits. SD cards use a similar interface.
2. Its controller also translates read/write requests into operations on the flash.
3. The host’s eMMC controller can use DMA to move data into RAM too.
4. eMMC traditionally handles one command at a time. eMMC 5.1 added a command queue of up to 32 tasks, still far fewer than NVMe.
5. The fastest mode (HS400) tops out around **400 MB/s**, and real sequential reads are often 100-300 MB/s.

### Comparison

|Interface|Bus|Command queues|Typical sequential speed|
|---|---|---|---|
|**NVMe**|PCIe|Up to 65,535 queues|3,500-10,000+ MB/s|
|**SATA SSD**|SATA (AHCI)|1 queue, 32 commands|~550 MB/s|
|**eMMC**|MMC, 8-bit parallel|1 command (32 tasks in eMMC 5.1)|100-400 MB/s|

On a board where the root filesystem is on eMMC and a separate NVMe drive is mounted elsewhere, keep large datasets and build outputs on the NVMe. See [Running the Orin: Storage](https://ricojia.github.io/2026/10/04/robotics-jetson-running-the-orin/#storage-work-on-nvme-not-emmc-smaller-and-slower) for the Jetson Xavier example.
