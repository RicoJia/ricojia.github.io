---
layout: post
title: Linux - Connecting to a Jetson Over Ethernet
date: 2018-01-05 13:19
subtitle: Finding the Jetson's IP address, and letting Docker containers reach it
comments: true
tags:
  - Linux
  - Jetson
---

## How To Find Jetson Xavier's Address Over Ethernet

My Jetson Xavier is connected to my laptop through Ethernet.

### 1. Find Your Computer's Ethernet Connection

```bash
ip -br addr
# or
ip addr
# or
ip a
```

- `ip` manages and displays Linux networking information.
- `-br` means brief output.
- `addr` shows the IP addresses assigned to your network interfaces.

An **interface** is a network connection, such as Ethernet or Wi-Fi. You might see:

```
enp0s31f6    UP      192.168.10.200/24
eth0         DOWN
wlp147s0     UP      ...
```

Here, `enp0s31f6` is your active Ethernet interface, and **`192.168.10.200` is your computer's address**, not the Jetson's. Names starting with `en` are Ethernet and `wl` are Wi-Fi; see [how interface names are built](https://ricojia.github.io/2018/01/27/linux-networking/#ethernet-and-nic).

The `/24` (CIDR notation) specifies the subnet: the first 24 bits identify the network. Devices on the same subnet can talk directly without a router. In this case:

| Address                         | Meaning                |
| ------------------------------- | ---------------------- |
| `192.168.10.0`                  | Network address        |
| `192.168.10.1`–`192.168.10.254` | Usual device addresses |
| `192.168.10.255`                | Broadcast address      |

So now we search `192.168.10.x`, **assuming the Jetson is configured on that same subnet**. Plugging in an Ethernet cable doesn't automatically guarantee matching IP settings. In my case the laptop's interface is set to NetworkManager's "Shared to other computers" mode (see [below](#letting-docker-containers-reach-a-jetson-over-ethernet)), which runs a DHCP server that hands the Jetson an address on this subnet.

### 2. Look For Devices On That Connection

First, check your computer's neighbor table:

```bash
ip neigh show dev enp0s31f6

```

I see:

```
192.168.10.175 lladdr 48:b0:2d:3a:9f:66 REACHABLE
```

Or for short, use the command `ip neigh`,

```
192.168.10.175 dev enp0s31f6 lladdr 48:b0:2d:3a:9f:66 DELAY 
192.168.1.230 dev wlp147s0 FAILED 
```

On an IPv4 Ethernet network, Linux uses **ARP** (Address Resolution Protocol) to ask:

> "Who has this IP address? Tell me your Ethernet hardware address."

The neighbor table caches those IP-to-hardware-address mappings (as shown above):

- **IP:** `192.168.10.175`
- **MAC address:** `48:b0:2d:3a:9f:66`, the Ethernet hardware address. The first three bytes identify the manufacturer, and `48:b0:2d` belongs to NVIDIA, which is a good hint that this is the Jetson.
- **`REACHABLE`:** Linux recently confirmed that neighbor was reachable

Then I logged onto this device and verified that it was the Jetson:

```bash
ssh <USER>@192.168.10.175
hostname
```

However, this table **isn't a complete inventory**. A device may be connected without having an entry yet, because entries only appear after your computer has talked to it. Next, actively probe the subnet by pinging every address in parallel:

```bash
for i in $(seq 1 254); do
    (
        ping -c 1 -W 1 "192.168.10.$i" >/dev/null 2>&1 &&
        echo "up 192.168.10.$i"
    ) &
done
wait
```

You might get:

```
up 192.168.10.200
up 192.168.10.175
```

`.200` is your own computer; `.175` is another device.

A device whose firewall drops pings won't show up in this loop. Two tools can scan more thoroughly:

- `sudo arp-scan --interface=enp0s31f6 --localnet` sends ARP requests on layer 2. ARP only works within the local subnet and only for IPv4, but a device can't usually ignore ARP and still communicate, so firewalls rarely hide it.
- `sudo nmap -sn 192.168.10.0/24` does host discovery without a port scan. On a local Ethernet subnet with `sudo`, nmap also uses ARP; across a router it falls back to ICMP pings and TCP probes on layer 3.

## Letting Docker Containers Reach a Jetson Over Ethernet

I connected a Jetson to my laptop over Ethernet:

```text
Docker container
      |
      v
hammurabi host
      |
      | enp0s31f6
      v
192.168.10.0/24 network
      |
      v
Jetson: 192.168.10.175
```

The host can ping the Jetson:

```bash
ping 192.168.10.175
```

But Docker containers could not.

The reason is that `enp0s31f6` is configured in NetworkManager as:

```text
IPv4 Method: Shared to other computers
```

In this mode, NetworkManager treats the Jetson network as a downstream LAN. It creates a firewall chain like:

```text
nm-sh-fw-enp0s31f6
```

This chain allows replies and traffic from the downstream network, but rejects new forwarded traffic going toward it.

A ping from the host works because it follows this path:

```text
host process -> OUTPUT -> enp0s31f6 -> Jetson
```

That does not hit the Linux `FORWARD` chain.

A ping from a Docker container follows this path:

```text
container -> docker bridge -> host FORWARD chain -> enp0s31f6 -> Jetson
```

That does hit the `FORWARD` chain, so NetworkManager blocks it.

## Quick Fix

Insert ACCEPT rules before NetworkManager’s reject rules:

```bash
sudo iptables -I nm-sh-fw-enp0s31f6 3 -s 172.17.0.0/16 -d 192.168.10.0/24 -o enp0s31f6 -j ACCEPT
sudo iptables -I nm-sh-fw-enp0s31f6 3 -s 172.19.0.0/16 -d 192.168.10.0/24 -o enp0s31f6 -j ACCEPT
sudo iptables -I nm-sh-fw-enp0s31f6 3 -s 172.20.0.0/16 -d 192.168.10.0/24 -o enp0s31f6 -j ACCEPT
sudo iptables -I nm-sh-fw-enp0s31f6 3 -s 10.233.0.0/24 -d 192.168.10.0/24 -o enp0s31f6 -j ACCEPT
```

These rules allow Docker bridge networks to reach the Jetson LAN.

## Make It Persistent

`NetworkManager` may regenerate its firewall rules when the interface reconnects, so manual `iptables` edits can disappear.

Create a dispatcher script:

```bash
sudo tee /etc/NetworkManager/dispatcher.d/99-docker-to-jetson-lan.sh > /dev/null <<'EOF'
#!/bin/bash

IFACE="$1"
STATUS="$2"

[ "$IFACE" = "enp0s31f6" ] || exit 0

case "$STATUS" in
    up|connectivity-change)
        for subnet in 172.17.0.0/16 172.19.0.0/16 172.20.0.0/16 10.233.0.0/24; do
            iptables -C nm-sh-fw-enp0s31f6 -s "$subnet" -d 192.168.10.0/24 -o enp0s31f6 -j ACCEPT 2>/dev/null \
                || iptables -I nm-sh-fw-enp0s31f6 3 -s "$subnet" -d 192.168.10.0/24 -o enp0s31f6 -j ACCEPT
        done
        ;;
esac
EOF

sudo chmod +x /etc/NetworkManager/dispatcher.d/99-docker-to-jetson-lan.sh
```

Important heredoc detail:

```bash
EOF
```

must start at the beginning of the line.

## Summary

The host could reach the Jetson because host-originated traffic does not go through `FORWARD`.

Docker containers could not reach it because container traffic is forwarded through the host, and NetworkManager’s shared-mode firewall rejected new forwarded traffic toward the Jetson LAN.

The fix is to explicitly allow Docker bridge subnets through the NetworkManager shared firewall chain.
