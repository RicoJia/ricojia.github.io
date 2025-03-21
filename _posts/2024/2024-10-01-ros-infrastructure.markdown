---
layout: post
title: ROS1 Infrastructure Notes
date: '2023-10-01 13:19'
subtitle: A Running List of ROS1 Infrastructure I found Useful
comments: true
tags:
    - ROS
---

## ROS Basics

### Where ROS binaries are stored

In ROS, many packages come in the C++ binary form. They are installed through `apt`, and are stored in `/opt/ros/noetic/lib`.

## ROS Actions

The ROS Action really is a mechanism that keeps track of the exeuction state of a task.  ROS Actions are built on top of ROS topics.

### Lifetime of ROS Actions

<p align="center">
<img src="https://github.com/user-attachments/assets/c1685270-7eb3-4cf7-a321-5fca50aa8b22" height="300" width="width"/>
</p>

1. Pending : The goal has been received by the action server, not yet processed
2. Active : The goal is currently being processed
3. Preempted : The goal has been preempted for a new goal
4. Succeeded : Goal has been successfully achieved
5. Aborted : Action Server encountered an internal error and has to abort the issue (similar to HTTP error code 503)
6. Rejected : Goal has been rejected without processing
7. Preempting : Goal is being cancelled after it's been active
8. Recalling : cancelling the goal before it got accepted
9. Recalled : Goal has been recalled

There is a topic `<ACTION_NAME>/status`. Its status reflects the status of the task:

```bash
status_list:
  - goal_id:
      id: "unique_goal_id_1"
    status: 0  # PENDING
    text: ""

status_list:
  - goal_id:
      id: "unique_goal_id_1"
    status: 1  # ACTIVE
    text: ""

status_list:
  - goal_id:
      id: "unique_goal_id_1"
    status: 3  # SUCCEEDED
    text: "Goal reached successfully."

status_list:
  - goal_id:
      id: "unique_goal_id_1"
    status: 4  # ABORTED
    text: "Goal was aborted."

status_list:
  - goal_id:
      id: "unique_goal_id_1"
    status: 2  # PREEMPTED
    text: "Goal was preempted by a new goal."
```
