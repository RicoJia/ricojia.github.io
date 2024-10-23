---
layout: post
title: RGBD SLAM - GPU Setup
date: '2024-08-18 13:19'
subtitle: Summary Of Nvidia Orin Nano Setup, Docker For General Machine Learning
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

## Coursera GPU Set Up

If you are a paid Coursera member, you can get free GPU access [in the Residual Networks assignment](https://www.coursera.org/learn/convolutional-neural-networks/home/week/3). As of Sept 25 2024, I can see:

```
%!nvidia-smi
Wed Sep 25 17:46:22 2024       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.05    Driver Version: 525.85.05    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A10G         On   | 00000000:00:1C.0 Off |                    0 |
|  0%   42C    P0   140W / 300W |  12940MiB / 23028MiB |     23%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

Here an `Nvidia A10G` is used. It's an industrial grade GPU: 150W, Ampere Architecture, It's 31.2 TF (FP32), and 250 TOPS (INT8).

**A huge pitfall is after a certain number of hours (1h?), the ipynb kernel is deactivated and you are no longer eligible to train.**

### SSL PITFALL

When I was trying to do

```python
from torchvision import models
model_ft = models.resnet18(weights='DEFAULT')
```

I saw:

```python
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth

---------------------------------------------------------------------------
SSLEOFError                               Traceback (most recent call last)
File /usr/lib/python3.8/urllib/request.py:1354, in AbstractHTTPHandler.do_open(self, http_class, req, **http_conn_args)
   1353 try:
-> 1354     h.request(req.get_method(), req.selector, req.data, headers,
   1355               encode_chunked=req.has_header('Transfer-encoding'))
   1356 except OSError as err: # timeout error
   ...
URLError: <urlopen error EOF occurred in violation of protocol (_ssl.c:1131)>
```

According to this post, it might be that Python version on the jupyter notebook does not support TLS1.1 +. I tried the solution there but to no avail. So for now, this notebook is **only good for Training from scratch, not for transfer learning.**

```python
import ssl
urlopen('https://www.howsmyssl.com/a/check', context=ssl._create_unverified_context()).read()
```

### Clean Up

At the end, let's be good citizens and free up memories for others:

```python
%%javascript
IPython.notebook.save_checkpoint();
if (confirm("Clear memory?") == true)
{
    IPython.notebook.kernel.restart();
}
```

## Wifi Pitfall

`network-manager` is a quite wonky. Sometimes the dhcp registration would suddenly drop on my Wifi, and ethernet doesn't work either. After a LOT of trial and error, this is the solution I came up with. It's quite painless, all you need to do is to paste this in `~/.bashrc`, source it, and type in the console `wifi_orin_connect`

```
wifi_orin_connect(){
    # if you see wpa issues, do wpa_passphrase <SSID> <PASSWORD> | sudo tee /etc/wpa_supplicant.conf
    
    set -ex

    sudo rfkill unblock wifi
    sudo ip link set wlan0 up
    sudo wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant.conf
    echo "this might take a while"
    # add google and cloudflare your DNS servers
    echo "prepend domain-name-servers 8.8.8.8 1.1.1.1;" | sudo tee -a /etc/dhcp/dhclient.conf
    sudo dhclient wlan0
}
```
