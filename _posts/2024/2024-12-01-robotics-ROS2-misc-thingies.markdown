---
layout: post
title: Robotics - [ROS2 Foundation] Ros2 Miscellaneous Thingies
date: '2024-11-30 13:19'
subtitle: Name Spacing
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