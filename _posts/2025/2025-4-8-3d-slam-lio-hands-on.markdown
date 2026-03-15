---
layout: post
title: Robotics - [3D SLAM - 5] LIO Hands-On
date: 2025-4-8 13:19
subtitle: Spatial Tiling
header-img: img/post-bg-o.jpg
tags:
  - Robotics
  - SLAM
comments: true
---

## Issues That I Encountered

### IMU / LiDAR Synchronization

There are several real-world timing problems to handle:

- The ROS publisher, bag writer, and LiDAR driver are all **bursty** — LiDAR data can arrive more than 10 seconds late, and IMU data may be missing for that same window.
- IMU messages can arrive **after** the LiDAR scan they should precede.
- We can always assume IMU messages eventually arrive in **chronologically increasing** order.

**Coping strategy** — double-buffer with triggering:

```
on every new IMU or LiDAR message:
    while lidar_buffer is not empty:
        lidar_msg = lidar_buffer.front()
        last_imu_ptr = find_imu_msgs_until(lidar_msg, imu_buffer)

        if last_imu_ptr == imu_buffer.begin():
            # No IMU coverage yet — fall back to lidar-only odometry
            print_warning(); break

        if last_imu_ptr == imu_buffer.end():
            # All buffered IMU may still be needed — wait for more
            break

        measurement = imu_buffer[begin .. last_imu_ptr] + lidar_msg
        imu_buffer.erase(begin .. last_imu_ptr)
        callback(measurement)
```

---

## Spatial Tiling

### 1. Configuration

| Parameter | Example | Description |
|---|---|---|
| `tile_size` | 5.0 m | Edge length of each spatial grid cell |
| `local_map_radius` | 30.0 m | Radius around the robot to include tiles |
| `tile_downsample_leaf` | 0.1 m | Voxel downsample leaf size per tile |

### 2. Tile Key Hashing

Each 3D position is discretized into an integer `TileKey{x, y, z}`. Keys are stored in an `unordered_map<TileKey, TileData, TileKeyHash>` where `TileData` holds a single accumulated `PCLCloudXYZIPtr`.

### 3. Accumulation (`add_new_keyframe`)

When a new keyframe is accepted:

1. The keyframe's **world-frame cloud** is appended (`+=`) to the tile for the keyframe pose.
2. If `tile_downsample_leaf > 0`, the tile is immediately **voxel-grid downsampled** in place to keep its point count bounded.
3. `rebuild_local_map` is called to refresh the active local map.

### 4. Local Map Rebuild (`rebuild_local_map`)

Triggered every time a keyframe is added (for flat-map scan matchers like `pcl_ndt` / `icp`):

1. Compute the `TileKey` of the **current robot position**.
2. Iterate over tiles within a cube of half-width `ceil(local_map_radius / tile_size)` around that center tile.
3. Include a tile if its center is within `radius + tile_size * 0.87` of the robot position. The factor $0.87 \approx \frac{\sqrt{3}}{2}$ is the circumscribed-sphere radius of a unit cube, ensuring no border tile is incorrectly excluded.
4. Concatenate all qualifying tile clouds into a **fresh `global_map_`**, replacing the old one.

### 5. Consumption (Scan Matching)

`rebuild_local_map` runs **asynchronously**. The caller retrieves the assembled local map via `get_map()` and passes it to the lidar odometer's scan matcher. A `future.wait()` guard at the top of the next `align()` call ensures the update completes before the next scan match begins.

### Summary Flow

```
New Keyframe
  → hash pose → accumulate into tile → per-tile voxel downsample
  → rebuild_local_map: radius query over tile grid → concatenate matching tiles
  → async update scan matcher target
  → next align() waits on future → scan match against fresh local map
```

The key benefit: instead of matching against an ever-growing global map, only tiles **spatially near the robot** are included, keeping the local map size bounded and scan matching fast — especially important on long trajectories.

### Why does reducing `tile_size` help?

Smaller tiles improve scan matching quality for two reasons:

1. **Tighter radius selection.** The local map is assembled by including all tiles whose centers fall within `local_map_radius`. With large tiles, a single border tile can contribute points far outside the intended radius, adding noise to the scan matcher's target cloud. Smaller tiles make the spatial boundary sharper — fewer irrelevant points pollute the local map.

2. **More effective per-tile downsampling.** Each tile is voxel-downsampled independently. A smaller tile covers a tighter spatial region, so the voxel grid produces a more spatially uniform point distribution within that region. With large tiles the same voxel grid is spread over a wide area, leaving some sub-regions over- or under-represented after downsampling.

The trade-off is a larger hash map and more tile-boundary bookkeeping, but for typical `tile_size` values (1–10 m) this overhead is negligible.
