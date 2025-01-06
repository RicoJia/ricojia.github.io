---
layout: post
title: Linux - SSH and X Window System
date: '2018-01-05 13:19'
subtitle: X Window System, SSH 
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

- Check login history: 
    - `who` display the last 3 logins
    - `last` display a longer list.

### SSH vs SSHD

- SSH (Secure Shell Client): Initiates machine as an SSH Client to connect to a remote server
  - `ssh username@remote_host 'ls -la /var/www'` : run commands on a remote server
  - `scp local_file username@remote_machine:PATH`
  - All data transmission here is encrypted.
- SSHD (Secure Shell Daemon):
  - Listens on a specified port (default port 22) for incoming SSH requests.

## What is X?

X Window System (or X11, X) renders graphics on display hardware. It can interact with input devices such as a mouse, keyboard, etc., to create effects like dragging windows and clicking. X is widely used in UNIX like systems: Linux, Solaris, etc.

X has an x server (screen) and x clients (keyboard, mouse, etc.). They talk to each other through network protocol, so a screen can display inputs remotely.

<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/24188326-ceec-4299-977f-75752a5626e9" height="400" alt=""/>
        <figcaption><a href="https://en.wikipedia.org/wiki/X_Window_System">Source: Wikipedia </a></figcaption>
    </figure>
</p>

## Be Careful With Adding XClients

If a child process does not have access to the X server, but is running as root (e.g., in a container), one **not super secure** solution is to allow any process running as root to have access to the X server.

```bash
xhost +local:root
```

- Note that this command cannot be run in a script because running a script opens up a new shell. It can be sourced, though (source executes the command in the current shell).

Alternatively,

```bash
xhost +local:
```

This allows a local user to access Xterminal

## What If I Can't See Images From Remote Machine?

1. Log onto remote server

2. `echo $DISPLAY` This should display your current X11 display, something like `localhost:10.0`

3. `xauth list` If nothing prints on console, it means ssh did not automatically generate the X11 authorization cookies on the local display properly.
    1. If you don't see this, first `sudo vim /etc/ssh/sshd_config`
    2. Make sure these exists:

        ```python
        X11Forwarding yes
        X11UseLocalhost yes
        ```

    3. Restart the SSH console:

        ```python
        sudo systemctl restart ssh
        ```

    4. Remove `.Xauthority`

        ```python
        rm ~/.Xauthority
        ```

    5. Log out and reconnect to generate X11 authorization cookies

        ```python
        ssh -Y rico@rico-orin
        ```

## Remote SSH Forwarding W/ AT&T Routers

1. Log onto the `AT&T` page, `192.168.1.254`
2. Go to Ip Passthrough -> change allocation mode from `Off` to `Passthrough`
3. Use DCHP Fixed. Under the device list, find the ssh server's MAC address 
4. In NAT/GAMING, add the device as an SSH server. Make sure the correct device is added.
5. Check IP addresses that have logged on, use `who`.

Trouble shooting

- `sudo tcpdump -ni any tcp port 22` run a packet capture on the server itself. 
