---
layout: post
title: Linux HodgePodge
date: '2024-05-06 13:19'
excerpt: This blog is a hodge-podge collection of Facts about Linux that I found useful
comments: true
---

This is a running hodge-podge list of Linux concepts and commands that I found useful in my robotics career. I will keep adding ingredients and spices here. 🍲

Enjoy!

## Services
A `systemd` service in unix has:

- A unit file that configures how a service should be started. Example:

    ```bash
    [Unit]
    Description=My Web Service

    [Service]
    ExecStart=/usr/bin/my-web-service
    ExecStop=/usr/bin/my-web-service-stop
    Restart=on-failure
    ```

    - Unit files are commonly stored in  `/lib/systemd/system`

### Common operations

- Enable and Disable
    - Enabling a service creates a symlink in `/etc/systemd/system` to the unit file. **So the service won't start immediately, but loaded and could autostart (depends on config) at next boot**
    - Disable: Deletes symlink in  `/etc/systemd/system`, but the unit file itself is fine
- Load
    - When your system is rebooted, services with symlinks in `/etc/systemd/system` are **loaded** (but not started yet)
    - To reload: 
        - `sudo systemctl daemon-reload`: scans newly modified/added unit files and load them into systemd configuration. **So use this when there are changes to unit file(s)**
        - `sudo systemctl reload <service>`: reloads a service manually
    - There is nooo unload
- Start & Stop
    - Start and stop services immediately. Systemd will execute commands specified in its configs to do these.
- mask
    - masking is to symlink the service's unit file to `/dev/null`. So even an enabled service cannot load the service.

### Autostart
Autostarting can depend on a few factors:
- There could be multiple targets: `multi-user.target`, or `graphical.target`. If they can't be reached at boot, then the service is not started. 
- Conditions. Unit files could have `Condition*` that must be met for service to start. E.g., `ConditionPathExists=/some/path` will autostart the service at boot only if the path exists. 


## Poll

`poll` is used to monitor changes on a specified file descriptor. E.g., checking if there's a new USB device being plugged in. This is also known as "event-driven-programming". In python: 

```python
import select
poller = select.poll()
fd = 0
poller.register(fd, select.POLLIN)  # checking input events
READ_PERIOD = 5 # in milliseconds
events = poller.poll(READ_PERIOD)
...
```

## X Window System (X)

### What is X?

X Window System (or X11, X) renders graphics on display hardware. It can interact with input devices such as a mouse, keyboard, etc., to create effects like dragging windows and clicking. X is widely used in UNIX like systems: Linux, Solaris, etc.

X has an x server (screen) and x clients (keyboard, mouse, etc.). They talk to each other through network protocol, so a screen can display inputs remotely.

<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/24188326-ceec-4299-977f-75752a5626e9" height="400" alt=""/>
        <figcaption><a href="https://en.wikipedia.org/wiki/X_Window_System">Source: Wikipedia </a></figcaption>
    </figure>
</p>

### Be Careful With Adding XClients
If a child process does not have access to the X server, but is running as root (e.g., in a container), one **not super secure** solution is to allow any process running as root to have access to the X server.

```bash
xhost +local:root
```

- Note that this command cannot be run in a script because running a script opens up a new shell. It can be sourced, though (source executes the command in the current shell).
