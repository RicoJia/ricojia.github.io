---
layout: post
title: Linux - Provisioning of My Machine
date: '2018-01-23 13:19'
subtitle: fzf
comments: true
tags:
    - Linux
---

## NVIDIA Driver

1. Open the `Additional Drivers application`
2. Select `NVIDIA driver metapackage from nvidia-driver-570 (proprietary, tested)`
3. Reboot
4. Run `nvidia-smi` to ensure it worked

## Nvidia Docker Container Toolkit

1. This toolkit allows nvidia drivers and docker containers to play nice.
    - NOTE: Do not perform the experimental packages step

2. [Follow NVIDIA's guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
    - don't forget the Configure Docker section at the end

3. Run sample docker to ensure it's working

    ```cpp
    sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi`
    ```

## Programming

### Linters

Here are the linters I generally use:

| Language   | Version | Linters/Analyzers               |
| ---------- | ------- | ------------------------------- |
| C++        | C++17   | cppcheck, cpplint, clang format |
| C          | C11     | cppcheck, cpplint, clang format |
| CMake      | 3.28.1  | lint\_cmake                     |
| Python     | 3.10.12 | flake8, pep257, pylint, mypy    |
| JavaScript | 14.x    | ESLint                          |
| Rust       | 1.72.1  | cargo test, cargo clippy        |
| Dockerfile | 25.0.2  | hadolint                        |
| XML        | –       | xmllint                         |
| YAML       | –       | yamllint                        |

## Optional

- fzf: Command line fuzzy finder

```
git clone https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install
```

    - `alt-c`: **built-in goto**, very useful
    - `ctrl-t`: copies file names over
    - `ctrl-r`: reverse history

- Navi: [Interactive command line cheat sheet tool](https://github.com/denisidoro/navi/releases/latest)
  - bring up navi (TODO: need configuration)
    - type something + `ctrl-g`
