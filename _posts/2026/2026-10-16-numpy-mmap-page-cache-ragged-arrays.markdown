---
layout: post
title: Deep Learning - Why 100 GB of NumPy Data Slowed Down Our ML Training
date: 2026-10-16 13:19
subtitle: NumPy Memory Mapping, Linux Page Cache, Swap, Pause and Resume, and Ragged Arrays
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Deep Learning
  - Machine-Learning
  - Linux
  - Python
---

## 1. The Problem

Suppose we're training a 3D LiDAR object recognition model to detect cars, pedestrians, and cyclists. We have three training configurations: centred crops, shifted crops, and temporal crops.

We launched three training jobs simultaneously, expecting the GPU to handle them efficiently. After 69 minutes:

|Training job|Progress|Speed|
|---|---|---|
|Centred|Epoch 7/60|~9 min/epoch|
|Shifted|Epoch 12/60|~5.5 min/epoch|
|Temporal|Not started|Reading at ~130 KB/s|

Our machine has 62 GB of RAM, and two training datasets occupy approximately 51 GB each. Initially, we suspected the SSD. However, the NVMe was reading at approximately 680 MB/s.

The actual problem was **memory pressure causing Linux to swap program memory onto a slow mechanical hard drive**. To understand why, let's start with the data format.

## 2. Why Does Each Dataset Occupy 51 GB?

Each training sample is a crop of a 3D point cloud. Each point contains four `float32` values:

$$
p_i=(x_i,y_i,z_i,I_i)
$$

where `x, y, z` are 3D coordinates and `I` is intensity. Different objects have different numbers of points:

|Object|Example point count|
|---|---|
|Pedestrian|650|
|Cyclist|1,400|
|Car|3,200|

Our original storage layout uses one `.npy` file per recording, containing **a structured array** with one row per crop (we use a solid-state LiDAR, so its point cloud is structured). Each row reserves space for 4,096 points, regardless of how many real points exist:

```
Crop A: 650 real points + 3,446 zeros
Crop B: 1,400 real points + 2,696 zeros
Crop C: 3,200 real points + 896 zeros
```

This is called **padding**. With 779,000 crops, four values per point, and four bytes per value:

$$
779000 \times 4096 \times 4 \times 4 \approx 51.1\text{ GB}
$$

We have two such datasets, totalling approximately 102 GB.

Importantly, `.npy` itself does not require 4,096 points. This is a constraint of our fixed-size structured array layout.

## 3. How Does NumPy Memory Mapping Work?

Normally, loading a NumPy array reads the whole array into memory:

```python
data = np.load("recording.npy")
```

Instead, we use **memory mapping (`mmap`)**:

```python
data = np.load("recording.npy", mmap_mode="r")
```

Rather than loading all 51 GB immediately, Linux maps the file into the process's virtual address space. When we access a crop with `crop = data[1000]`, Linux:

1. Checks whether the required file pages are already in RAM.
2. If not, reads them from the SSD.
3. Makes those pages available to the process.
4. Keeps them in RAM for possible reuse.

The cached file data is called the **page cache**.

### Why is page cache useful?

Training repeatedly reads the same data across epochs. If a crop remains in page cache, subsequent reads avoid SSD access.

However, page cache consumes physical RAM. Our machine was observed with approximately 57 GB of page cache out of 62 GB of total RAM. Two 51 GB datasets cannot both remain entirely cached. That's where memory pressure begins.

## 4. What Is the Difference Between Page Cache and Swap?

Linux manages two important categories of memory:

| |Page cache|Swap|
|---|---|---|
|Contains|File-backed data|Evicted memory pages, mainly anonymous memory|
|Example|Memory-mapped `.npy` data|Python buffers, CPU training state|
|Can Linux discard it?|Yes, if clean|No, must preserve the data|
|Read back from|Original file|Swap device|

**Page cache does not overflow into swap.** When RAM becomes scarce, Linux can reclaim clean cached file pages because their contents already exist on disk. But program memory that has no file backing cannot simply be discarded; eligible pages may instead be written to swap.

```
RAM
├── Page cache: LiDAR crops
├── Python program memory
├── DataLoader buffers
└── Other system memory
```

When the training processes continuously load large datasets, Linux must choose which memory to reclaim. In our case, some program memory was moved to swap on a mechanical hard drive.

## 5. Why Did Training Stall? Memory Thrashing

The observed sequence was:

1. Multiple jobs read large `.npy` datasets.
2. Linux fills much of RAM with page cache.
3. Memory pressure causes some program memory to be swapped out.
4. Training processes need that memory again.
5. Linux must read it back from a slow hard drive.
6. The cycle repeats.

This is called **memory thrashing**: the machine spends excessive time moving memory instead of computing. Our measurements showed:

|Measurement|Observed value|
|---|---|
|Total RAM|62 GB|
|Page cache|~57 GB|
|Swap used|4.2 GB|
|Swap location|7,200 RPM HDD|
|HDD read throughput|~4.6 MB/s|
|I/O pressure (`full`)|~70%|

The temporal process was particularly affected. It had approximately 1.27 GB swapped out and was blocked waiting for memory-related I/O.

### Why is HDD swap particularly slow?

Mechanical hard drives are poor at random reads. Suppose one small random read takes 10 milliseconds, and each read retrieves a 4 KiB page:

$$
\text{Reads per second}=\frac{1}{0.01}=100
\qquad\Rightarrow\qquad
100\times4096\approx0.41\text{ MB/s}
$$

This is an illustrative serial random-read calculation, not the measured HDD throughput. Even though the NVMe dataset drive was fast, the training processes could be blocked by much slower swap accesses.

### Immediate solutions

- **Reduce concurrency:** run the temporal training job separately.
- **Move swap to NVMe:** avoid slow mechanical HDD swap I/O.
- **Lower swappiness:** for example, `sudo sysctl vm.swappiness=10` encourages reclaiming file cache relative to swapping anonymous memory.
- **Disable swap where appropriate:** avoids swap stalls, but accepts a greater risk of out-of-memory termination.

These changes address memory management. We can also reduce the dataset itself.

## 6. Why Are We Wasting Almost Half the Storage?

Recall that every crop reserves 4,096 points, but our average crop contains only approximately 2,177 real points. The average padding fraction is:

$$
\frac{4096-2177}{4096}\times100\approx46.9\%
$$

Nearly half of the point storage is unnecessary padding. The most important observation is that **our neural network never uses those padded points**. The existing loader already does something like:

```python
count = crop["point_count"]

xy = crop["xy"][:count]
height = crop["height"][:count]
intensity = crop["intensity"][:count]
```

It discards padding and rasterizes the real points into a fixed-size grid, such as `(3, 48, 48)`. The CNN receives this grid, not the original 4,096-point array. Therefore, we can eliminate padding from disk without changing the model input.

## 7. Solution: Ragged Arrays with Two `.npy` Files

A **ragged array** stores variable-length samples without padding. Instead of storing every crop in a fixed-size block, we concatenate all real points into one continuous array, using two files per recording:

|File|Contents|
|---|---|
|`recording.npy`|Crop metadata, including `point_start` and `point_count`|
|`recording.points.npy`|All real points concatenated into one `(N, 4)` array|

### Example

Suppose three crops contain different numbers of points:

|Crop|Points|`point_start`|`point_count`|
|---|---|---|---|
|Pedestrian|A, B, C|0|3|
|Car|D, E, F, G, H|3|5|
|Cyclist|I, J|8|2|

The points file contains:

```
Index:   0 1 2 3 4 5 6 7 8 9
Points:  A B C D E F G H I J
```

To retrieve the car crop:

```python
points = np.load("recording.points.npy", mmap_mode="r")
metadata = np.load("recording.npy", mmap_mode="r")

crop = metadata[1]
start = int(crop["point_start"])
count = int(crop["point_count"])

car_points = points[start:start + count]   # points D through H
```

No padding is required. Both files remain memory-mappable, so Linux reads only the required file pages rather than loading the whole dataset.

## 8. How Much Space Do We Save?

Previously, every crop stored 4,096 points. With ragged storage, we store approximately 2,177 real points per crop on average. At 16 bytes per point:

$$
S_{\text{old}}=779000\times4096\times16 \approx51.1\text{ GB},
\qquad
S_{\text{new}}=779000\times2177\times16 \approx27.1\text{ GB}
$$

Adding approximately 0.5 GB for metadata gives:

|Metric|Original|Ragged|
|---|---|---|
|Points stored per crop|4,096|~2,177 average|
|Storage per dataset|~51.1 GB|~27.6 GB|
|Two datasets|~102.2 GB|~55.2 GB|
|Padding|Yes|No|
|Memory mapping|Yes|Yes|

That's approximately a **46% reduction in storage per dataset**, and less data for Linux to read and cache. However, two 27.6 GB datasets still consume substantial cache space on a 62 GB machine, so swapping and memory contention remain possible.

## 9. Why Two Files Instead of One?

A normal `.npy` file stores one NumPy array. Our metadata and concatenated points are different arrays. There are alternatives:

|Format|Partial reads|Tradeoff|
|---|---|---|
|Two `.npy` files|Memory mapping|Simple, standard NumPy|
|`.npz`|No direct NumPy mmap|Combines arrays into one archive|
|HDF5|Efficient dataset slicing|Additional dependency and file-handle management|
|Custom binary|Memory mapping|Requires custom implementation|

We chose two `.npy` files because they're simple and preserve our memory-mapped loading workflow. We also avoid creating one file per crop, which would mean approximately 779,000 files per dataset.

## 10. What Changes in the Training Pipeline?

The network architecture doesn't need to change:

```
Original pipeline                 New pipeline

3D point cloud                    3D point cloud
      ↓                                 ↓
Extract crop                      Extract crop
      ↓                                 ↓
Pad to 4,096 points               Store real points + offsets
      ↓                                 ↓
Store in .npy                     Load real points using mmap
      ↓                                 ↓
Load and discard padding          Rasterize into fixed-size grid
      ↓                                 ↓
Rasterize into fixed-size grid    Neural network
      ↓
Neural network
```

The 4,096-point limit can remain as a maximum at crop creation time; crops exceeding that size are subsampled. Shorter crops no longer need padding at any stage.

Implementation requires updating the crop builder, dataset loader, and viewer, followed by testing that old and new representations produce equivalent model inputs.

## 11. Pausing and Resuming Training with `kill -STOP` and `kill -CONT`

Another way to reduce concurrency is to temporarily pause a training job to free CPU resources for another one, without losing the progress of existing jobs. Linux allows us to **pause and resume a process without restarting it**.

### Pausing a training process

```bash
kill -STOP <PID>
```

Here, `PID` is the process ID. The `SIGSTOP` signal tells Linux to suspend the process:

- The process **stops executing instructions.**
- Its RAM and allocated **GPU memory remain reserved**.
- Its model weights, optimizer state, and current training position remain intact.
- It does not need to reload a checkpoint when resumed.

For example, a training job paused halfway through epoch 20 can continue from that position.

A PyTorch `DataLoader` with `num_workers > 0` runs its workers as separate processes, so stopping only the main PID leaves them running until their prefetch queues fill. To pause the whole job, signal its process group instead, e.g. `kill -STOP -- -<PGID>` (find the PGID with `ps -o pgid= -p <PID>`).

### Resuming the process

```bash
kill -CONT <PID>
```

The `SIGCONT` signal allows the process to execute again, continuing from where it was suspended. Unlike terminating and restarting training, no training progress is intentionally discarded.

**Important:** pausing does not release GPU memory, and the process's state is preserved only while it remains alive. A reboot or process termination still loses uncheckpointed progress. Pausing does not free RAM either: under memory pressure, Linux may swap a stopped job's memory out (Section 4), and resuming it then means reading that memory back from swap.

To continue, send `SIGCONT` to the paused processes with `kill -CONT <PID>`. This is useful when coordinating multiple jobs on a shared GPU machine, provided enough GPU memory remains available for the jobs that continue running.
