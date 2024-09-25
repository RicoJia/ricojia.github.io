---
layout: post
title: RGBD SLAM - Setting Up Nvidia Orin Nano
date: '2024-08-18 13:19'
subtitle: Summary Of Nvidia Orin Nano Setup And Docker For Machine Learning
header-img: "img/post-bg-unix"
tags:
    - RGBD Slam
    - CUDA
    - Deep Learning
comments: true
---

## Preparation

- Jetson Orin Nano Developer Kit (8GB)
- MicroSD Card (128GB)
    - My microSD card's read and write speed can reach up to 140MB/s. An NVMe SSD could be 1500MB/s or more. So **try with an SSD if speed has become a bottle neck**
- **A data capable USB-C cable**
- Complete datasheet (very lengthy, feel free to skip)
    - [Jetpack SDK](https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/index.html) Jetpack SDK includes accelerated software libraries, APIs, sample applications, developer tools and documentation.

## Successful Attempt (SD Card)

I was primarily following [this video](https://www.youtube.com/watch?v=q4fGac-nrTI). 

1. Use SDK Manager: `https://developer.nvidia.com/sdk-manager`
2. Put the board in recovery mode (by connecting pin `GND` to `FC_REC` using a female-female jumper wire to connect)
3. Start SDK manager. The flash might fail. If it fails, go to step 4, otherwise, go to step 5
4. `cd ~/nvidia/nvidia_sdk/JetPack_6.0_Linux_JETSON_ORIN_NANO_TARGETS/Linux_for_Tegra && sudo ./flash.sh jetson-orin-nano-devkit mmcblk0p1`. This flashed nvidia linux succssfully onto my machine
5. Configure linux on the board
6. `sudo apt update && sudo apt upgrade && sudo apt install nvidia-jetpack`
7. `dpkg -l | grep nvidia-jetpack` should see version like `6.0-bXXX`
8. [Optional] A diagnostic tool `jtop` is recommended but optional: `sudo apt install python3-pip && sudo -H pip3 install -U jetson-stats`
9. [Optional] `jtop` and go to tab 7. The only missing item should be OpenCV.

## Successful Attempt (SSD)

[Implementation Reference](https://zhuanlan.zhihu.com/p/697333833)

1. I have [crucial SSD](https://www.amazon.com/dp/B0B25LQQPC?ref=ppx_yo2ov_dt_b_fed_asin_title)
2. Follow the [latest installation guide](https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/IN/QuickStart.html#to-flash-the-jetson-developer-kit-operating-software)
3. Choose the NVMe option
    ```bash
    sudo ./tools/kernel_flash/l4t_initrd_flash.sh --external-device nvme0n1p1 \
      -c tools/kernel_flash/flash_l4t_t234_nvme.xml -p "-c bootloader/generic/cfg/flash_t234_qspi.xml" \
      --showlogs --network usb0 jetson-orin-nano-devkit internal
    ```
4. Common Issues
    - If you see this, reconnect the USB and power of the board.
        ```
        Error: Unrecognized module SKU
        Error: failed to generate images
        Cleaning up...
        ```
    - If you see `nfs timeout`,
        ```
        Waiting for device to expose ssh ......Waiting for device to expose ssh ...Run command: flash on fc00:1:1:0::2
        SSH ready
        mount.nfs: Connection timed out
        Flash failure
        Either the device cannot mount the NFS server on the host or a flash command has failed. Debug log saved to /tmp/tmp.HVseonQElu. You can access the target's terminal through "sshpass -p root ssh root@fc00:1:1:0::2" 
        ```
        - Then we need to install nfs and change the firewall Setting
            ```
            ## 安装nfs
            sudo apt update
            sudo apt install nfs-kernel-server

            ## 防火墙打开nfs端口
            sudo ufw status
            sudo ufw allow from fc00:1:1:0::2 to any port nfs
            sudo ufw allow from fc00:1:1::/48 to any port nfs
            ```
5. If `nvcc` is not a recognized command, add below to `~/.bashrc`:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Machine Learning Docker Setup

To fully unleash the power of an Nvidia Jetson Machine for machine learning, there are multiple ways to set up the environment. A common way is to set up conda. In my application though, I prefer to use an Docker image so I can hook up my ML application with ROS / ROS2 applications. To do that, we need to set up a CUDA image properly, then select the correct pytorch version.

Luckily, [dusty-nv and others have created a nice Github package to streamline this process](https://github.com/dusty-nv/jetson-containers). This package is a small Docker build system that creates the final desired Docker image through multi-stage builds (like a chain). E.g., one can build an image with `pytorch`, `ROS Noetic` and `Jupyterlab`.

If you don't want much hassle like me, this could be a good starting point. But just in case you are curious, here is the place to [check for the Pytorch version](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)


## Failed Attempt For Nvidia Orin Nano Setup

I tried below but the script did NOT WORK on my machine. When running `./flash_jetson_external_storage.sh` I see

```
cp: cannot stat '/home/rico/Downloads/bootFromExternalStorage/R35.4.1/Linux_for_Tegra/rootfs/lib/modules/5.10.120-tegra/kernel/drivers/mtd': No such file or directory
```

1. Download [Jetson Linux and sample file system](https://developer.nvidia.com/embedded/jetson-linux)
2. Check the assumptions under [Preparing a Jetson Developer Kit for Use](https://developer.nvidia.com/embedded/jetson-linux). The host system is your own laptop, the target is the Nano.
3. `git clone https://github.com/jetsonhacks/bootFromExternalStorage.git`, this is a set of helper scripts.

- Note this is downloading `R35.4.1` of Nvidia Linux. So replace all its instances with the current version: `for file in *; do sed -i 's/'R35.4.1/R36.3.0/g "$file"; done`
