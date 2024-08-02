---
layout: post
title: Linux - Udev Rules And Systemd Services
date: '2018-01-07 13:19'
excerpt: Udev Rules Systemd Services
comments: true
---

## Udev

 Unix Kernel is a monolithic kernel and provides API to access these hardware. Udev is a device manager of Linux Kernel, and serves as a successor of the Unix Kernel. It handles device nodes in `/dev` and raises user space events (uevent) when hardware devices are added to the system. In 2012, it's moved to systemd

### Udev Work flow

1. Udev finds a device through:
    - name
    - vendor id (vid)
    - pid (device)
    - Model name
2. Assign names to devices
3. Create symlinks
4. Trigger scripts

Udev rules end in `.rules`. They are in two places:

- `etc/udev/rules.d/`: user's custom udev rules.
- `/usr/lib/udev/rules.d/`: System udev rules

If there are udev rules with same names in these two directories, the custom one will take precedence.

udev rules are processed based on the number before their names:

- `50-udev-default.rules` will be processed before `60-...`

A Udev config file example (Intel RealSense D415):

```bash
# Sets USB device permission to 0666 (read and write) on device with the specified vendor id and product id. Then execute the script in /usr/local/...
SUBSYSTEMS=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0a80", MODE:="0666", GROUP:="plugdev", RUN+="/usr/local/bin/usb-R200-in_udev"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0a66", MODE:="0666", GROUP:="plugdev"
# Intel RealSense recovery devices. Assigns the device into the plugdev group
SUBSYSTEMS=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ab3", MODE:="0666", GROUP:="plugdev"

# Applies to devices with kernel name matching iio*. It adds file permission to read, write, execute for everyone (0777).
# Then, runs the command chmod -R 0777 /sys/%p to set file permission in /sys for this device.
KERNEL=="iio*", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad5", MODE:="0777", GROUP:="plugdev", RUN+="/bin/sh -c 'chmod -R 0777 /sys/%p'"

# Applies the hid_sensor_custom driver on the specified device. HID subsystem is for mouse, keyboard, sensors, etc.
# HID exposes sensor data through Industrial I/O (IIO) interface. 
DRIVER=="hid_sensor_custom", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad5", RUN+="/bin/sh -c 'chmod -R 0777 /sys/%p && chmod 0777 /dev/%k'"
```

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