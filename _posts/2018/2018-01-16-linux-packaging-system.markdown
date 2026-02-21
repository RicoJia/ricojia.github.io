---
layout: post
title: Linux - Packaging Systems
date: '2018-01-16 13:19'
subtitle: apt, dpkg
comments: true
tags:
    - Linux
---

## Installation Chain

`apt`, `apt-get`, `apt-cache` talks to the repository servers, download `.deb`, calculate dependency tree, the hand the final files to `dpkg` for the actual install. E.g.,

```bash
sudo apt install curl
```

- apt finds curl & its deps
- downloads .debs
- dpkg -i each_downloaded.deb

### dpkg

`dpkg`, the debian package manager, works directly  with `.deb` files. It maintains a database of Debian packages.

- `dpkg -i <FILE>.deb`: install `.deb` file
- `dpkg -r foo`: remove (remove the binaries but keeps config files)
- `dpkg -P foo`: purge (remove the binaries + configs)
- `dpkg -S <Path>`: search for `path` of the file. `dpkg -S /usr/bin/python3 → python3.8: /usr/bin/python3`
- `dpkg -l`: Shows version, description of all deb packages
