---
layout: post
title: Linux Miscellaneous Thingies
date: '2023-06-17 13:19'
subtitle: SSH
comments: true
tags:
    - Linux
---

## SSH

- Generate an SSH Key: `ssh-keygen -t rsa -b 4096`. 
    - This will create `~/.ssh/id_rsa` and optionally sets a passphrase
- Copy ssh onto a remote machine: `ssh-copy-id username@remote_host`
    - You will be prompted for the password of the remote machine. All your public keys will then land in `~/.ssh/authorized_keys`.

- `sudo nmap -sn 192.168.1.0/24`: uses ICMP echo requests (ping), TCP (SYN) packets on OSI layer 3 (the network layer). This is more robust than `sudo arp-scan -l` because the latter uses ARP (Address Resolution Protocol) protocol on layer 2 (the local subnet). Some devices may not respond due to its firewall settings. Also, ARP is an IPv4 protocol. IPv6 devices may also avoid using it. 