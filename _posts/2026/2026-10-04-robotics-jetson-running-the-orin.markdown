---
layout: post
title: "Robotics - [Jetson 2] Running the Orin: Power Modes, Benchmarking, and Pitfalls"
date: '2026-10-04 00:00'
subtitle: Power modes, sustained performance, memory reporting, Wifi, and how I burned a board
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Jetson
    - CUDA
    - Embedded
comments: true
---

This is part 2 of my Jetson notes. Part 1, [Flashing the Orin Nano](https://ricojia.github.io/2024/08/18/rgbd-slam-setup-nvidia-orin-nano/), covers getting JetPack onto the board.

## I Burned My $500 Nvidia-nano-orin

This happened when I connected it to a Waveshare Rover with its "Jetson Nano Adapter"

- How could that happen? My suspicions are: I put them on 4 metal pillars for stability. They might have caused a short on local components? **So never put your precious orin or any boards on metal surfaces**
- I connected the orin to the robot (ESP32) through its Jetson Nano Adapter after it was powered on. Then I turned on the robot. 10s later, I saw a black smoke coming from under the NVidia board
  - **Orin might have been subjected to a high inrush current when powered on the robot.** Always turn on the main power, let the voltage stablize and establish ground voltage, no hot plugging.

## Power Modes and Sustained Performance

Jetson Orin’s inference performance depends on its power mode, clock settings, and cooling. A benchmark should record all three so its results can be reproduced.

### Power Modes and Clock Settings

Jetson uses **dynamic voltage and frequency scaling (DVFS)** to adjust operating frequencies according to workload and operating constraints. Two tools control its performance settings:

|Tool|Purpose|
|---|---|
|`nvpmodel`|Selects a power mode that defines power budgets, available CPU cores, and frequency limits.|
|`jetson_clocks`|Sets CPU, GPU, and memory clocks to their maximum permitted frequencies within the selected mode.|

To inspect the current power mode and enable maximum permitted clocks:

```bash
sudo nvpmodel -q
sudo jetson_clocks
```

This is **not overclocking**: it stays within NVIDIA’s supported operating limits. Overclocking means operating beyond the manufacturer’s specified frequencies.

### Cooling Determines Sustained Performance

Maximum clock settings can improve throughput when dynamic clock scaling was limiting performance. However, they do not guarantee sustained maximum speed. Thermal and power protections remain active and can reduce clocks during a long workload. See [NVIDIA’s power and performance documentation](https://docs.nvidia.com/jetson/archives/r35.6.5/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html).

The following **illustrative example—not measured data or documented throttling thresholds—**shows how throughput could decline as a board heats up:

|Elapsed time|GPU temperature|GPU clock|Throughput|
|---|--:|--:|--:|
|0:00|48°C|918 MHz|42 FPS|
|1:00|58°C|918 MHz|42 FPS|
|2:00|66°C|918 MHz|42 FPS|
|3:00|73°C|918 MHz|41 FPS|
|4:00|79°C|765 MHz|36 FPS|
|5:00|82°C|612 MHz|31 FPS|

A benchmark report should therefore include a time series of temperature, actual clocks, and throughput. A fixed five-minute run alone does not establish thermal stability; the measurements should show whether temperature and performance have settled.

### Reporting Memory Usage

Jetson AGX Orin **has no dedicated GPU VRAM.** Its **CPU and integrated GPU share the same physical LPDDR5 memory pool**. GPU allocations therefore consume system memory rather than a separate graphics-memory bank. [NVIDIA’s CUDA for Tegra application note](https://docs.nvidia.com/cuda/archive//11.4.4/pdf/CUDA-for-Tegra-AppNote.pdf) explains this shared-memory architecture.

In benchmark tables, use **“Device memory footprint”** instead of “VRAM,” and specify what was measured: GPU allocations, process memory, or total system memory usage. These measurements are not interchangeable.

A suitable footnote is:

> Jetson AGX Orin uses shared LPDDR5 memory for the CPU and GPU. The reported footprint represents the stated memory measurement, not dedicated VRAM usage.

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
