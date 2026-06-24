---
layout: post
title: Robotics - Isaac SIM setup
date: '2024-08-25 13:19'
subtitle: Isaac SIM
header-img: "img/post-bg-unix"
tags:
    - Robotics
comments: true
---

## Installation

Route 1: Native Execution

- Install ROS 2 natively on your system.
- Set up the ROS 2 context and add a ROS 2 subscriber block.
- Ensure Fast DDS is properly configured for ROS 2 communication.
- Clone and integrate the following workspace repository: isaac-sim/IsaacSim-ros_workspaces.

Additional Notes:

- For headless installation, refer to the Isaac Sim headless container guide.
- For development and deployment, explore Isaac Sim's start-dev and deployment workflows. A helpful walkthrough is available here.
- See the official ROS bridge documentation for integration details.

1. verify GPU access inside Docker:

```bash
docker run --rm --runtime=nvidia --gpus all \
  nvcr.io/nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

## 2. Pull Isaac Sim container

```bash
docker pull nvcr.io/nvidia/isaac-sim:6.0.0
```

NVIDIA’s current container docs use `nvcr.io/nvidia/isaac-sim:6.0.0`. ([Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html "Container Installation — Isaac Sim Documentation"))

## 3. Create persistent cache folders

Do this once:

```bash
mkdir -p ~/docker/isaac-sim/cache/main
mkdir -p ~/docker/isaac-sim/cache/computecache
mkdir -p ~/docker/isaac-sim/config
mkdir -p ~/docker/isaac-sim/data
mkdir -p ~/docker/isaac-sim/logs
mkdir -p ~/docker/isaac-sim/pkg
mkdir -p ~/.cache/ov/hub

mkdir -p ~/isaac_projects/caliper_sdg
mkdir -p ~/isaac_projects/assets
mkdir -p ~/isaac_projects/output

sudo chown -R 1234:1234 ~/docker/isaac-sim ~/.cache/ov/hub ~/isaac_projects
```

The cache mounts matter because first launch compiles/warmups lots of shaders; keeping the cache makes later runs much faster. NVIDIA’s docs also warn that stale cache mounts can cause config or launch problems, in which case removing/recreating `~/docker/isaac-sim` can help. ([Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html "Container Installation — Isaac Sim Documentation"))

## 5. Run compatibility checker inside container

```bash
docker run --entrypoint bash -it --gpus all --rm --network=host \
  -e "ACCEPT_EULA=Y" \
  -v ~/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
  -v ~/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw \
  -v ~/.cache/ov/hub:/var/cache/hub:rw \
  -v ~/isaac_projects:/workspace:rw \
  -u 1234:1234 \
  nvcr.io/nvidia/isaac-sim:6.0.0 \
  ./isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window
```

You want to see something like:

```text
System checking result: PASSED
```

NVIDIA’s docs explicitly mention this expected pass message. ([Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html "Container Installation — Isaac Sim Documentation"))

Note: `ACCEPT_EULA=Y` is required to accept the container license. `PRIVACY_CONSENT=Y` is optional telemetry opt-in; I would omit it unless you intentionally want to opt in. NVIDIA documents both environment variables. ([Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html "Container Installation — Isaac Sim Documentation"))

Here in my setup I encountered:

```
GPU 0: VRAM [not enough]  
total: 8.59 GB  
minimum: 10 GB
```

But I expected potential crashes and tried this anyways:

```bash
docker run --name isaac-sim --entrypoint bash -it --gpus all --rm --network=host \
  -e "ACCEPT_EULA=Y" \
  -v ~/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
  -v ~/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw \
  -v ~/.cache/ov/hub:/var/cache/hub:rw \
  -v ~/isaac_projects:/workspace:rw \
  -u 1234:1234 \
  nvcr.io/nvidia/isaac-sim:6.0.0
```

The path is:

STL
  -> convert to USD
  -> load USD into Isaac stage
  -> attach semantic label: motor_sleeve
  -> create camera + light + table
  -> BasicWriter saves RGB + 2D bbox

If STL is probably in **millimeters**. So in Isaac, we probably want scale = 0.001. Then,

```
cp "$HOME/Downloads/J3 motor_sleeve.STL" ~/isaac_projects/assets/motor_sleeve.stl
```
