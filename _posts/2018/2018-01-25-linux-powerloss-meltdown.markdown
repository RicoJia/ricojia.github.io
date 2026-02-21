---
layout: post
title: Linux - Power Loss Meltdown
date: '2018-01-25 13:19'
subtitle: initramfs
comments: true
tags:
    - Linux
---

## 🚑 What broke

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/755Dww8f/initframfs.jpg" height="300" alt=""/>
    </figure>
</p>
</div>

1. Sudden power-loss left the system in an inconsistent state. (while `/roota` was being modified)

    1. First boot stopped in BusyBox with `ALERT! UUID=255a5309-… does not exist`
    2. `initramfs` could not see the NVMe root partition.
        - `initramfs` is "initial RAM-file-system". It runs as the very first userspace code that the kernel starts—before your real root disk is mounted.
        - It's a tiny, compressed cpio archive (usually /boot/initrd.img-…) unpacked directly into RAM. It contains a minimal Linux userland (BusyBox), essential kernel modules, and scripts in /init.
        - Functionalities:
            - Loads drivers for storage, USB, encryption, LVM, etc.
            - Locates and mounts the real root filesystem by UUID/label.
            - Switches the kernel’s root from the RAM disk to your SSD/NVMe.
            - Hands control to `/sbin/init (systemd)`.
        - How `initramfs` is built:
            - `update-initramfs` gathers drivers listed in `/etc/initramfs-tools/modules`, plus what `mkinitramfs` can auto-detect, then packs them into one file per installed kernel.

2. Solution:
    1. Booted Live-USB, ran `blkid`, confirmed `root = /dev/nvme0n1p2` with correct UUID.
    2. Ran `fsck.ext4 -yf /dev/nvme0n1p2`.
        - `fsck` = File System Check.
        - `ext4` partitions the helper binary is `fsck.ext4`
    3. Chrooted into the install and forced modules into `initramfs`:

        ```
        nvme nvme_core usbhid hid_generic xhci_pci xhci_hcd. 
        ```

    4. Added `rootdelay=10` temporarily to grub?
        - grub is a bootloader. A bootloader takes care of getting the operating system started up. It is also responsible for allowing the user to select between multiple operating systems at boot
    5. Normally, we need to use the keyboard in BusyBox to type recovery commands. However, it was dead. It turned out that system76 computers need to follow [this official guide](https://support.system76.com/articles/bootloader/). When running as root through `chroot`

        ```
        apt install --reinstall linux-image-generic linux-headers-generic
        update-initramfs -c -k all
        update-grub
        exit

4. After we fixed storage detection, Ubuntu reached the purple GNOME “Oh no!” screen

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/CR90KXSJ/oh-no.jpg" height="300" alt=""/>
    </figure>
</p>
</div>

5. `gnome-shell` or `GDM` crashed because the NVIDIA kernel module and user libraries were mismatched.
    - Re-installing the Nvidia driver gave a blue background with a blinking cursor—GDM started but never showed the greeter.
        1. Purged every old NVIDIA package & module (`apt purge '^nvidia-.*', modprobe -r …`)
        2. Re-installed one clean version (`sudo ubuntu-drivers autoinstall` → driver 570).
        3. Enrolled MOK for Secure Boot.
            - MOK (Machine Owner Key) is a personal signing key you enroll so Secure Boot will allow third-party kernel modules like the NVIDIA driver.
        4. One can use `nvidia-smi` to verify that the nvidia driver is successfully installed

6. With LightDM in place the greeter appeared, but logging in looped straight back because of stale caches / permissions in $HOME.

    1. Booted with `systemd.unit=multi-user.target` (at Grub modification page. Get into it by pressing `<F8>`) to guarantee a console.
    2. Installed and activated LightDM: `apt install lightdm; systemctl disable gdm3; systemctl enable lightdm`.
        - LightDM and GDM (system default) are Display manager
        - LightDM is a quick life-raft when GDM / Wayland misbehave—you can keep it installed as a fallback.

7. Fix login loop
    1. Freed / checked space, ensured / and `$HOME` were read-write.
    2. Re-owned files: `sudo chown -R $USER:$USER $HOME`.
    3. Reset GNOME caches: `mv ~/.cache ~/.cache.bak, mv ~/.config/dconf ~/.config/dconf.bak`, etc.
    4. Completed pending upgrades: `apt full-upgrade`.

8. Confirm desktop
    1. Logged in via LightDM on Xorg → desktop works.
    2. Optional future step: reinstall/enable GDM once everything is stable.
