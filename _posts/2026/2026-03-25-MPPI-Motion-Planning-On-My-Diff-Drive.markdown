---
layout: post
title: MPPI And Motion Control On My Diff Drive
date: 2026-03-25 13:19
subtitle: Remote Claude Code Setup
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine-Learning
---

## Motor Control

So, tested on carpet, I think it's about not having (+/-) on the spot that makes it

- motor controller
 	- aware that the robot turns better using something like:

left  = ±u
right = 0

rather than opposite-wheel spinning. So we want

```
Global planner
    |
    v
Path / waypoints
    |
    v
Local planner / MPPI
    - samples feasible motor commands
    - predicts robot motion
    - scores path tracking, obstacle clearance, heading, smoothness
    |
    v
Motor command filter / safety layer
    - clamps commands
    - forbids bad command pairs
    - rate limits changes
    - applies deadband compensation
    |
    v
Serial motor driver
```

### MPPI TODO

Need to test with feasible motion.

That is command: (+-0.8, 0). I noticed that the right wheel was rolling.

MPPI samples  commands closer to what the robot actually accepts:

```
u = [left_motor_cmd, right_motor_cmd]

ACTIONS = [
    # forward / backward
    (+0.2, +0.2),
 ... 

    # maybe forbid these if bad
    # (+0.4, -0.4),
    # (-0.4, +0.4),
]
```

Then MPPI samples from these commands and predicts motion using your learned or measured dynamics model. Your controller no longer says:

> I want `omega = 1.5 rad/s`, so I command `(+u, -u)`.

Instead it says:

> I want to reduce heading error, and among the feasible actions, `(left=+0.45, right=0.0)` produces the best progress.

## Step 1 - Build Feasibility Velocity Set

```

```

1. Collect command-response data per surface.
 1. For each surface: carpet, wood, heavy_carpet, left_cmd  ∈ [-0.5, 0.5]right_cmd ∈ [-0.5, 0.5], with stride=0.1. That gives121 commands.
 2. For each pair:
  1. Stop robot
  2. Wait 0.5 s
  3. Apply left_cmd, right_cmd for 4 s
  4. Record odom/IMU/encoder data in CSV
  5. Stop robot
  6. Wait 1 s
 3. Repeat 3–5 times

```
timestamp  
surface_label  
left_cmd  
right_cmd  
battery_voltage  
x  
y  
yaw  
v_measured  
w_measured
```

2. Aggregate the info

```
surface  
left_cmd  
right_cmd  
mean_v  
mean_w  
std_v  
std_w
num_trials
```

3. Build inverse map:  surface, desired_v, desired_w → best_left_cmd, best_right_cmd

```
surface = heavy_carpet
desired_v = 0.0
desired_w = 0.4

best command:
left_cmd = +0.6
right_cmd = 0.0
```

4. Build forward map:

```
surface = heavy_carpet
left_cmd = +0.5
right_cmd = -0.5

predicted:
v = 0.01 m/s
w = 0.04 rad/s
success_score = 0.2
```

5. Runtime surface estimator:  commanded wheels + observed motion → surface probability
 1. for the past 1s, predict what surface
 2. Then output a probability score: `P(surface) = surface_score / sum(surface_scores)`
6. Feedforward:  desired_v,w + surface probability → better wheel command
7. MPPI
 1. soft switching:  `predicted_motion =    0.7 * carpet_model(cmd)  + 0.2 * wood_model(cmd)  + 0.1 * heavy_carpet_model(cmd)`
 2. Then during run time, given v and w, and wheel command, the controller infers which surface it's on (and this will help with the feedforward term too). controller Publishes  surface on a topic, mppi reads it, then switches feasibility set.

## Surface Switching

### Use a Ros2 Topic for fast changing runtime state
