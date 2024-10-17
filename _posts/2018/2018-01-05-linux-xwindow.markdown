---
layout: post
title: Linux - X Window System
date: '2018-01-05 13:19'
subtitle: X Window System, Window Focus
comments: true
tags:
    - Linux
---

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

## Window Focus (Which seems to be an )

On my Ubuntu 22.04 system, when trying to download webpages / images from Chrome / Firefox, I noticed that the download dialog box is not focused in the first place. After searching for auto-focusing, I came up with a sub-optimal solution: **auto-shift to window that the mouse currently hovers on.**

1. `sudo apt install gnome-tweaks`
2. `Press the Super (Windows) key and type Tweaks, then press Enter.`
3. Choose `focus on hover`, 
    - If you don't want to see the other window that's out of focus already, choose "Raise Window When Focused"
4. Press Alt + F2, type r, and press Enter. **This restarts the GNOME Shell without logging out.**

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
