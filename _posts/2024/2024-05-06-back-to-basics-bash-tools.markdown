---
layout: post
title: Going Back To The Basics - Bash Tools
date: '2024-05-06 13:19'
excerpt: This blog is a hodge-podge of bash commands that I found useful
comments: true
---

This is a running hodge-podge list of bash concepts and commands that I found useful in my robotics career. I will keep adding ingredients and spices here. 🍲

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


