---
layout: post
title: Robotics - [ROS2 Foundation] Ros2 Parameters
date: '2024-11-22 13:19'
subtitle: 
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

## ROS2 Parameters Introduction

[What are ROS2 parameters?](https://docs.ros.org/en/humble/Concepts/Basic/About-Parameters.html)

ROS2 Parameters are one means to change certain values in a node during runtime. They are associated initialized, set, and retrieved at the level of individual nodes, and hence their lifetimes are the same as their nodes' lifetimes.

Each parameter has **a key, value, and a descriptor**.

- The key is a string
- The value is one of the following types: `bool, int64, float64, string, byte[], bool[], int64[], float64[] string[]`
- The descriptor is a parameter's description

[Here is an example of how we can use them](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Parameters/Understanding-ROS2-Parameters.html):

1. Initialize / declare
    - Command Delare params?

        ```python
        from rclpy.node import Node
        class ParameterNode(Node):
            def __init__(self):
                super().__init__('parameter_node')
                self.declare_parameter('shared_value', 42.0)
        ```

    - When a parameter needs to take in different types, use `ParameterDescriptor` with the dynamic_typing member variable set to true

        ```python
        from rcl_interfaces.msg import ParameterDescriptor
        descriptor = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter('dynamic_param', 0, descriptor)
        ```

    - In CLI:
        - `ros2 param list` shows all parameters:

            ```
            /teleop_turtle:
                qos_overrides./parameter_events.publisher.depth
                use_sim_time
            ```

2. Get

- `param_value = self.get_parameter_or(param_name, None)`
- For parameters that can't be known ahead of time, they can be instantiated as  `allow_undeclared_parameters = True`

        ```python
        class MyNode(Node):
            def __init__(self):
                super().__init__('my_node', allow_undeclared_parameters=True) 
            def get_dynamic_param(self, param_name):
                param_value = self.get_parameter_or(param_name, None)
                return param_value
        def main(args=None):
            ...
            node.get_dynamic_param('undeclared_param')  # Should print a warning and return None
        ```

- In CLI:
  - `ros2 param get <node_name> <parameter_name>`
    - `ros2 param list`: a CLI python helper that pings a running node to retrive their parameters. It calls `list_parameter` service of each node.
    - So there could be a an exception like `Exception while calling service of node 'my_node': None` where the node is busy

  - View all params of a node, and they can be saved into an yaml:

            ```
            ros2 param dump <node_name>
            ros2 param dump /turtlesim > turtlesim.yaml
            ```

- Set.
  - In code?

        ```python
        from rclpy.parameter import Parameter
        class MyNode(Node):
            def update_param(self, key, val):
                param = Parameter(key, Parameter.Type.INTEGER, val)
                self.set_parameters([param])
        ```

  - In CLI:
    - `ros2 param set <node_name> <parameter_name> <value>`
    - Load from a yaml file:

            ```
            ros2 param load <node_name> <parameter_file>
            ```

    - Load parameters on start up:

            ```
            ros2 run <package_name> <executable_name> --ros-args --params-file <file_name>
            ```

## Rclcpp Parameters

### Overriding ROS 2 Parameters

You can expose and override parameters in two parts:

1. Inside your node (`RunImuNode`): in the constructor, **declare** each parameter (with an optional default) and **read** it into a member:

```cpp
class RunImuNode : public rclcpp::Node
{
public:
  explicit RunImuNode(const rclcpp::NodeOptions &opts = {})
    : Node("run_imu_node", opts)
  {
    // 1) Declare parameters (here with defaults)
    this->declare_parameter<bool>("is_millig",      true);
    this->get_parameter("is_millig",      is_millig_);
  }

private:
  bool        is_millig_;
};
```

2. From “user” or test code: when you instantiate the node, supply overrides via NodeOptions:

```cpp
rclcpp::NodeOptions options;
options.parameter_overrides({
  {"is_millig",            false},
});

auto run_imu_node = 
  std::make_shared<RunImuNode>(options);
```

3. **Overriding parameters can be done in ros launch level**

```
launch file:
    my_ws:
        ros__parameters:
            execute_service: ~/execute
#cpp
rcl_interfaces::msg::ParameterDescriptor execute_service_desc;
execute_service_desc.name = ros_toolbox::create_parameter_name(node_, "execute_service");
execute_service_desc.type = rclcpp::ParameterType::PARAMETER_STRING;
execute_service_desc.description = "Service name to execute capacities on.";
std::string execute_service = node_->declare_parameter(execute_service_desc.name, "~/capacity/execute", execute_service_desc);
```

In this example, the capacity name will ultimately become `~/execute`, instead of `~/capacity/execute`

### Passing `--ros-args -p <ARG>` to a ROS 2 GTest

You can make your test binary accept ROS 2 parameters (e.g. `visualize`) without touching your node by:

1. **Providing a custom `main()`**: Capture the full `argc/argv` (including `--ros-args`) and use them for both `rclcpp::init()` and Google Test:

```cpp
#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

// Store all CLI args so we can forward them later
static std::vector<std::string> g_test_args;

int main(int argc, char ** argv)
{
 ::testing::InitGoogleTest(&argc, argv);
 // Copy original args (including --ros-args)
 g_test_args.assign(argv, argv + argc);

 rclcpp::init(argc, argv);
 int ret = RUN_ALL_TESTS();
 rclcpp::shutdown();
 return ret;
}
```

2. Forwarding the ROS 2 args into your node. In your test fixture’s `SetUp()`, pass along `g_test_args` via `NodeOptions::arguments()`, then declare and read your test-only parameter:

```cpp
std::vector<std::string> g_test_args;
class TestIntegrateIMU : public ::testing::Test {
protected:
  bool visualize_{false};

  void SetUp() override {
    // 1) Build options with the exact same CLI args
    rclcpp::NodeOptions options;
    options.arguments(g_test_args);
    run_imu_node_ = std::make_shared<RunImuNode>(options);
    // 4) Declare & read the test-only “visualize” flag
    run_imu_node_->declare_parameter("visualize", false);
    run_imu_node_->get_parameter("visualize", visualize_);
    // …
  }
};

```

3. Run test with: `./test_integrate_imu --ros-args -p visualize:=true`

## Parameter Callback

In ROS 2, a parameter callback is a user-provided function that the middleware will invoke whenever someone tries to change one of your node’s parameters. It gives you a chance to:

- Validate the new value (e.g. range check, type check).
- Accept or reject the change (by returning success or failure).
- Modify the value before it’s actually set.

Under the hood:

```
ros2 param set /my_node my_param 42
```

The client issues a request to `/my_node/set_parameters`. Before the node actually writes `my_param = 42` into its local storage, it:

1. Runs all of your registered parameter callbacks in the order they were added.
2. Collects each callback’s rcl_interfaces::msg::SetParametersResult (in C++) or SetParametersResult (in Python).
3. If any callback vetoes (i.e. returns successful = false), the entire parameter update is rejected and none of the new values take effect.
4. Otherwise, the new values are committed and—only then—the node may publish a ParameterEvent for everyone listening.

```cpp
callback_handle_ =
    this->add_on_set_parameters_callback(
    std::bind(&MyNode::validate_params, this, std::placeholders::_1));

rcl_interfaces::msg::SetParametersResult
validate_params(const std::vector<rclcpp::Parameter> & params) {
rcl_interfaces::msg::SetParametersResult result;
result.successful = true;  // default: allow
for (auto &p : params) {
    if (p.get_name() == "my_param") {
    int v = p.as_int();
    if (v < 0 || v > 100) {
        result.successful = false;
        result.reason = "my_param must be in [0..100]";
    }
    }
}
return result;
}

// Keep the handle alive so the callback stays registered:
OnSetParametersCallbackHandle::SharedPtr callback_handle_;
```
