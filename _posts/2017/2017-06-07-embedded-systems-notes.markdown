---
layout: post
title: Electronics - Embedded Systems Notes
subtitle: HAL
date: '2017-06-01 13:19'
header-img: "img/bg-material.jpg"
tags:
    - Electronics
---

## HAL vs SAL

Hardware Abstraction Layer (HAL) is software that abstracts hardware-specific details from the application or higher-level software. It provides a unified API for accessing hardware functionalities, making it easier to develop software that can run on different hardware platforms without modification. Some examples include:

- GPIO, UART, I2C, etc.
- In Operating Systems, like Linux and Windows, HAL provides APIs for interacting with hardware without dealing with low-level device registers. E.g., HAL in Android, Linux kernel HAL, STM32 HAL

SAL (Software Abstraction Layer)

SAL is a layer that abstracts specific software functionalities or services, providing a common interface for applications to interact with system services, middleware, or third-party software components. Some examples include:

- Cloud Computing: Abstracting APIs for cloud services across different providers.
- Middleware Systems: Abstracting communication between applications and underlying libraries.