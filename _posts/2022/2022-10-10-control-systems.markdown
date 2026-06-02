---
layout: post
title: Control System Notes
date: 2022-10-11 13:19
subtitle: slew rate limit
comments: true
header-img: img/home-bg-art.jpg
tags:
  - Deep Learning
---

---

## Slew Rate Limit

A **slew limit** caps how quickly a command is allowed to change from one control loop update to the next. For example, `effort = slew_limit(effort)` prevents the requested effort from jumping instantly to a new value; instead, the effort delta is constrained to `+/- max_delta` each cycle. This is useful in control systems because sudden command changes can excite mechanical vibration, saturate actuators, cause current spikes, or make the system feel jerky. Slew limiting does not change the final target effort, but it shapes the transition so the controller output ramps toward the target at a controlled rate.
