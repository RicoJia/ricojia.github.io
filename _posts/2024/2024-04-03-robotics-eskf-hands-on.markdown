---
layout: post
title: Robotics - General ESKF Hands-on Test Notes
date: '2024-04-03 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
---

## Primers

- Did you turn on the message debugging flag, or are you compiling with `PRINT_DEBUG_MSGS`?
- Are you sure you are launching the right test?

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

## GPS Debugging Notes

- When GPS heading is invalid, are you omitting GPS messages?
  - One may choose not to omit GPS. We could just set their ESKF jacobians to 0.
- GPS needs IMU to be initialized
- IMU integration needs the first GPS to be found

## Data synchronization

- Do you get messages exactly the same timing?
- Are data synchronization, like the GPS and IMU Data sync correct?
- When GPS starts becoming valid, do you choose the pose there to be your initial pose?
  - Initial velocity is from the IMU. They are not directly corrected by GPS. So wrong initial velocity could make IMU spin a lot while its cartesian pose is around the GPS point
