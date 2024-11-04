---
layout: post
title: Linux Miscellaneous Thingies
date: '2018-01-17 13:19'
subtitle: Window Focus, Screenshots
comments: true
tags:
    - Linux
---

## Screenshots

- `sudo apt install gnome-screenshot`
- Open `keyboard shortcuts -> add custom shortcuts`
- I use `ctrl-alt-4` to enable `gnome-screenshot -a`

## Window Focus (Which seems to be an )

On my Ubuntu 22.04 system, when trying to download webpages / images from Chrome / Firefox, I noticed that the download dialog box is not focused in the first place. After searching for auto-focusing, I came up with a sub-optimal solution: **auto-shift to window that the mouse currently hovers on.**

1. `sudo apt install gnome-tweaks`
2. `Press the Super (Windows) key and type Tweaks, then press Enter.`
3. Choose `focus on hover`,
    - If you don't want to see the other window that's out of focus already, choose "Raise Window When Focused"
4. Press Alt + F2, type r, and press Enter. **This restarts the GNOME Shell without logging out.**

