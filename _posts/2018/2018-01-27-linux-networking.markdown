---
layout: post
title: Linux - Networking
date: '2018-01-27 13:19'
subtitle: NIC
comments: true
tags:
    - Linux
---

## Ethernet and NIC

NIC = Network Interface Controller; The ethernet card is by far the most common type of NIC. It's often integrated on the motherboard, but equivalently, it's equivalent to going through the PCIe switches. On laptops, there are usually 1 ethernet ports (and an associated ether card), on desktops, there could be 2.

- PCIe = Peripheral Component INterface Express, a high-speed interface for external components to talk to the motherboard. Common components include: GPU, NVMe SSD, Network Cards, etc.

Since Ubuntu 16.04, predictable network-interface was introduced. An ethernet NIC is 

```
en[p]<bus-number>s<slot-number>[f<function>]
└─┬──┘ └──┬─────┘
  │        └─── “device 0 in PCI slot 3”
  └──────────── “Ethernet” (as opposed to wl… = Wi-Fi, lo = loopback)
```

So `enp3s0` means “Ethernet adapter located on PCI bus 3, slot 0, function 0”.  On another PC you might see names like `enp5s0` or `eno1`. The NIC can be configured to an IP `192.168.1.xxx/24`. 

    - A subnet mask is the `/24`. Devices with the same subnet mask can talk directly without a router
    - `/24 is the ‘CIDR notation`

### Case Study: Connecting Livox Mid 360

https://i.postimg.cc/RZr9PNSP/2025-06-29-14-16-38.png

All Livox Mid- & Horizon-series sensors come with a fixed IP in the `192.168.1. range (/24 mask)`.
1. Upon connection, LiDAR negotiates speed/duplex w/ NIC
2. LiDAR will start publishing UDP packets.

The configuration steps include:

- Checking for the NIC name:

```
$ ip a
eno1: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc mq state DOWN group default qlen 1000
    link/ether cc:28:aa:45:68:bd brd ff:ff:ff:ff:ff:ff
    altname enp8s0
```

- Connect the device
- Checking if the physical link from the livox is up - `NIC Link is Up 100 Mbps Full Duplex` shows it's up now
```
ricojia@system76-pc:~/Downloads/LivoxViewer2 for Ubuntu v2.3.0$  sudo dmesg | grep -i -e eno1 -e enp8s0  
[    1.812914] igc 0000:08:00.0 eno1: renamed from eth0
[80870.168221] igc 0000:08:00.0 eno1: NIC Link is Up 100 Mbps Full Duplex, Flow Control: RX
```

- To confirm auto-negotiation is valid: 

```
sudo ethtool eno1  # ^--- look for "Link detected: yes/no", negotiated speed/duplex
    ...
	Supports auto-negotiation: Yes
```

- Disable ufw: `sudo ufw disable` (Livox Viewer2 would not be able to get point clouds otherwise)
