---
layout: post
title: Robotics - ROS2 Tests
date: '2024-11-30 13:19'
subtitle: colcon test
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---


## Use cases

### Passing ROS Arguments to ROS2 Test Executables

When writing integration tests in ROS2, you may need to pass command-line parameters `(--ros-args -p <param>:=<value>)` to your test binaries. By default, GTest’s main() intercepts all arguments, so ROS2 parameters aren’t automatically applied. Follow these steps to forward ROS arguments into your tests:

1. Provide a Custom `main()`. Override GTest’s default entry point and initialize both GTest and rclcpp with the original arguments:

```cpp
#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <string>

// Store all of main()’s argv into a global for later
static std::vector<std::string> g_test_args;

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  g_test_args.assign(argv, argv + argc);
  rclcpp::init(argc, argv);
  int ret = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return ret;
}
```

2. Forward Arguments in Your Test Fixture. In your fixture’s `SetUp()`, pass the captured arguments into `NodeOptions` so your node picks up any `--ros-args` parameters:

```cpp
void SetUp() override {
  rclcpp::NodeOptions options;
  options.arguments(g_test_args);
  options.parameter_overrides({
    ... 
  });
  node_ = std::make_shared<rclcpp::Node>("test_node", options);

  // Declare and retrieve your own parameter
  node_->declare_parameter("visualize", false);
  node_->get_parameter("visualize", visualize_);
}
```

3. Run Your Test with ROS Args. Finally, invoke your test binary with: `./test_integrate_imu --ros-args -p visualize:=true`
