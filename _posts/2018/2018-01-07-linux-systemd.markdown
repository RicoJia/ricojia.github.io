---
layout: post
title: Linux - Systemd Services
date: '2018-01-07 13:19'
excerpt: Systemd Services
comments: true
---

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