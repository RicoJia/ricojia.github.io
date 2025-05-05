---
layout: post
title: Robotics - [ROS2 Foundation] Ros2 Composition Instead Of Nodelets
date: '2024-11-26 13:19'
subtitle: Components
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

[Reference](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Composition.html#id2)

The basic idea of ROS2 composition is to have a unified API in place of ROS1 nodes and nodelets. Nodelets in ROS1 are compiled into shared libraries and can be loaded into one process during runtime through the tool chain. This way, nodelets can achieve zero copy in communication. The ROS2 composition mechanism is to achieve that too.

For example, in a SLAM system, we can have two component `FrontEnd` and `BackEnd`. We can dynamically load them into the same `SLAM` process and achieve zero-copy IPC (on the RMW/DDS middleware layer)

1. These two components are built into shared libraries. 
2. They subclass `rclcpp::Node`, and can launch their own topics, timers, etc. 
3. Then, they are registered using a macro (from the package rclcpp_components) so they are discoverable for runtime-loading.

## Basic Structure

### `talker.cpp` and Its `CMakeLists.txt`

```cpp
#include <rclcpp/rclcpp.hpp>

namespace composition
{

class Talker : public rclcpp::Node
{
public:
  explicit Talker(const rclcpp::NodeOptions & options)
  : Node("talker", options)
  {
    publisher_ = this->create_publisher<std_msgs::msg::String>("chatter", 10);
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      [this]() {
        auto msg = std_msgs::msg::String();
        msg.data = "hello from composable talker";
        publisher_->publish(msg);
      });
  }

private:
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace composition

// register with class loader — makes “composition::Talker” discoverable
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(composition::Talker)
```

- `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.5)
project(my_package)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)

add_library(talker_component SHARED
  src/talker_component.cpp
)

# link in ROS 2 dependencies
ament_target_dependencies(talker_component
  rclcpp
  rclcpp_components
  std_msgs
)

# register the component (this expands to RCLCPP_COMPONENTS_REGISTER_NODE)
rclcpp_components_register_nodes(talker_component
  "composition::Talker"
)

# install rules
install(
  TARGETS talker_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_package()
```
    - `rclcpp_components_register_nodes` must appear after add_library and ament_target_dependencies.
    - You only register each class once, even if you have multiple RCLCPP_COMPONENTS_REGISTER_NODE lines (one per class).

### `container_manager.cpp`

```cpp
// src/SLAM.cpp
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/component_manager.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // In-process container that will hold all your components
  auto container = std::make_shared<rclcpp_components::ComponentManager>(
    rclcpp::NodeOptions{});

  // Load FrontEnd (talker) into the container
  container->load_component(
    "my_package",                    // library name (package name by default)
    "composition::Talker"            // fully-qualified class name
  );

  // (Optionally) load more components here:
  // container->load_component("my_package", "composition::Listener");
  // container->load_component("other_pkg", "other_namespace::BackEnd");

  rclcpp::executors::StaticSingleThreadedExecutor exec;
  exec.add_node(container);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
```

- `CMakeLists.txt`

```c
add_executable(slam_container src/SLAM.cpp)

ament_target_dependencies(slam_container
  rclcpp
  rclcpp_components
)

# Ensure the container links against all component libraries
# so the symbols are available at load time.
target_link_libraries(slam_container
  talker_component       # if in same package
  # other_component_lib   # list any other component libs you intend to load
)

install(
  TARGETS slam_container
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
```
