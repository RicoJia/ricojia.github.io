---
layout: post
title: Linux - X Window System
date: '2018-01-05 13:19'
excerpt: X Window System
comments: true
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
