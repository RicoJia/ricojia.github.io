---
layout: post
title: Robotics - IMU Hands-On
date: '2024-04-02 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
---

## IMU tools

Velocity drift observed. Here, I'm proposing a two-phase static test to confirm true sensor drift:

- Initialization—standstill for N seconds to estimate biases.
- Static logging—remain stationary and record velocity.
- If velocity continues to grow, it’s genuine drift.
- Initial results show the same drift pattern—next step is to connect directly to the IMU board (bypass middleware) to rule out software filtering or fusion artifacts.

I observed that angular rate remains stable over the static test; no significant gyro drift detected.

Accelerometer bias

- Sampling at 33 Hz, the mean linear-acceleration bias is ≈ 0.3 m/s² (≈ 0.03 g).
    - Projected drift: ~560 m over one minute—still within typical MEMS accelerometer error bounds.
    - Changing the static initialization window shifts the estimated accelerometer bias (ba) and gyro bias (bg), which alters the trajectory shape but does not eliminate the drift or induced rotation.

- Bias characteristics
    - IMU biases are not fixed constants. They:
        - Drift with temperature.
        - Exhibit bias-random-walk (BRW) that grows proportionally to √t.
        - Often follow a 1/f noise spectrum rather than pure white noise.
    - Tactical in-run bias stability is on the order of micro-g, but bias repeatability is typically larger—see Advanced Navigation’s IMU introduction for details.

