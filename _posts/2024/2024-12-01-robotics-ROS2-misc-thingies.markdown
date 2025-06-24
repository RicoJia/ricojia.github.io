---
layout: post
title: Robotics - [ROS2 Foundation] Ros2 Miscellaneous Thingies
date: '2024-11-30 13:19'
subtitle: Name Spacing, Weird Issues
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

## Namespacing

If you have something like:

```cpp
class FiveCardStudPlayer : public rclcpp::Node {
public:
    FiveCardStudPlayer() : Node("five_card_stud_player"){
    ...
    }}
```

And you launch 2 instances of `five_card_stud_player`, ROS2 Humble **won't** crash the nodes. Instead, it will have two nodes with the colliding names

```
/five_card_stud_dealer
/five_card_stud_player
/five_card_stud_player    ← collision!
```

You can address this issue by using `NodeOptions`

```cpp
class MyNode : public rclcpp::Node {
public:
  MyNode(const rclcpp::NodeOptions & opts)
  : Node("my_node", opts)
  { /* … */ }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto args = rclcpp::remove_ros_arguments(argc, argv);
  rclcpp::NodeOptions opts;
  opts.arguments(args);

  auto node = std::make_shared<MyNode>(opts);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
```

- `std::make_shared<rclcpp::Node>(DEFAULT_NAME, opts)` will automatically change the name of the node
- `node->get_name()` will reflect the change
- Then you can call it with `ros2 run card_deck_game five_card_stud_player --ros-args --remap __node:=<unique_name>`

## Weird Issues

- You could have two instances of the same launch files of the same action server running. They could throw this error

    ```bash
    [capacity_manager_node-1] [ERROR] [1750698354.440384729] [commander.capacity_manager.rclcpp_action]: unknown result response, ignoring...
    ```

  - In ROS2, you are able to see two nodes with the **exact same namespace and name**. Not sure if that's a design choice
    - Across multiple processes you can launch two nodes with the same fully-qualified name and namespace; the DDS discovery layer just sees two matching participants.
    - This could be deadly when there's one dead but not-cleaned node. You might still see its topics but they are dead. Is there a way to avoid that?
      - If a node dies without calling `rcl_shutdown` (e.g. hitting SIGKILL), its DDS participant is never disposed cleanly.
        - A "node" dies just means `A single rclcpp::Node / rclpy.node.Node object`.
          - All publishers, subscriptions, services, actions, timers that belong only to that node are destroyed.
        - The process dies when `rclcpp::shutdown()` is called.
      - Most DDS vendors keep endpoints in the graph until a liveliness timeout expires (default is 20 s – 1 min, vendor-specific). During that window:
        - `ros2 node list` and `ros2 topic list` still show the dead participant.
