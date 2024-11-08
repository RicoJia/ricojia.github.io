---
layout: post
title: Linux Miscellaneous Thingies
date: '2018-01-17 13:19'
subtitle: Window Focus, Screenshots, File Differences
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

## File Differences

- `diff` shows which files are different and which files show in both places

```bash
rico@rico-Latitude-5420:~/file_exchange_port $ diff -qr filter_projects/ Fun_Projects/filter_projects/
Only in filter_projects/: build_project.sh
Files filter_projects/face_tracker/face_tracker.cpp and Fun_Projects/filter_projects/face_tracker/face_tracker.cpp differ
Files filter_projects/face_tracker/face_tracker.hpp and Fun_Projects/filter_projects/face_tracker/face_tracker.hpp differ
```

- If in a git repo, see when the directory was last modified:

```bash
rico@rico-Latitude-5420:~/file_exchange_port/Fun_Projects (master)$ git log -1 --format="%ci" -- filter_projects/
2022-03-31 12:07:13 -0500
```
- `-1`: Limits the output to the most recent commit.
- `--format="%ci"`: Formats the output to display the commit date in ISO format.
