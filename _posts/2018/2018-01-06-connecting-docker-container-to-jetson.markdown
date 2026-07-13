---
layout: post
title: Linux - Connecting Docker Container To Jetson Over Ethernet
date: 2018-01-05 13:19
subtitle: X Window System, SSH, Static IP
comments: true
tags:
  - Linux
---
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
