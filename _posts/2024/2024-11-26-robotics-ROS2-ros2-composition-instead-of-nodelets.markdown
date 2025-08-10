---
layout: post
title: Robotics - [ROS2 Foundation] Ros2 Composition Instead Of Nodelets
date: '2024-11-26 13:19'
subtitle: Components, Zero-Copy Intra-Processing-Communication (IPC)
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

### Loading & Unloading

- [Node Composition is only available in C++](https://docs.ros.org/en/humble/The-ROS2-Project/Features.html)
- To load / unload, there are two ways
  - Do it on the CLI [(reference)](https://www.youtube.com/watch?v=PD0VJYBkkfQ):

    ```
    ros2 run rclcpp_components component_container_mt 
    ros2 component load /ComponentManager card_deck_game card_deck_game::ComposableFiveCardStudDealer
    ricojia@system76-pc:~/file_exchange_port$ ros2 component list 
    /ComponentManager
        1  /composable_five_card_stud_dealer

    # Then
    ros2 component unload /ComponentManager 1
    ```

  - Use hidden services: [(reference)](https://design.ros2.org/articles/roslaunch.html)
    - `~/_container/load_node`
    - `~/_container/unload_node`
    - `~/_container/list_nodes`
  - `Component_manager` **won't reset index after unloading a component. So one must increase index for `unload <index>`**

- [QUIRK] If component container does not show the new components, try below:
  - `ros2 node list`
    - You might see a list of ghost entries, because ctrl-c stops the container process, but it does not instantly erase all DDS participants from the ROS2 graph. By default, Cyclone DDS waits for the participant's lease to expire (a few secs).
  - Do these to remove the ghost DDS participants:

        ```
        pkill -f component_container_mt
        ros2 daemon stop
        ros2 daemon start
        ```
  
### Component Containers are Executors

- component_container runs a `SingleThreadedExecutor` 
- component_container_mt runs a `MultiThreadedExecutor` 
- component_container_isolated runs an executor per component (CLI flag chooses single- vs multi-thread)

## Intra Process Comm (IPC) And Woes

ROS 2’s intra-process transport lets us avoid one copy by handing a `unique_ptr` (or `shared_ptr`) directly from publisher to subscriber—but it still allocates each message once. This is called "zero-copy IPC". In ROS2, there are two flavors of zero-copy: message loaning and intra-process unique-ptr handoff.

- Intra-process unique_ptr hand-off skips DDS serialization and the extra copy `from publisher buffer → RMW buffer → subscriber buffer`, **but it still requires memory allocation for messages**
- Message loaning on the other hand, uses a pre-allocated buffer that belongs to the middleware's shared memory pool. So no memory allocation for messages.

Currently (Nov 2024), ROS2 Humble middleware, `FastDDS` does not support message loaning. Consider switching to `CycloneDDS` which does support it.

Below, we demonstrate a card-game example where there are a dealer and multiple players. The dealer listens to `/show_hand` and publishes `/winner` as inter-process topics, and publishes onto `/hand` as an intra-process topic.

```cpp
class ComposableFiveCardStudDealer : public rclcpp::Node {
  public:
    ComposableFiveCardStudDealer(rclcpp::NodeOptions options) : Node("composable_five_card_stud_dealer", options.use_intra_process_comms(true))
    

        // disable intra-process-comm because this is inter-process-comm
        rclcpp::SubscriptionOptions no_intra_process_sub_opts;
        no_intra_process_sub_opts.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
        // Subscriber for players showing their hands:
        show_sub_ = create_subscription<ShowHand>(
            "show_hand", 10,
            std::bind(&ComposableFiveCardStudDealer::on_show_hand, this, std::placeholders::_1),
            no_intra_process_sub_opts);
        
        // disable intra-process-comm because this is inter-process-comm
        rclcpp::PublisherOptionsWithAllocator<std::allocator<void>> no_intra_process_pub_opts;
        no_intra_process_pub_opts.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
        // Now create the “winner” publisher with transient_local durability:
        winner_pub_ = this->create_publisher<std_msgs::msg::String>(
            "winner",
            rclcpp::QoS(1).reliable().transient_local(),
            no_intra_process_pub_opts);
    
        // TODO: Is this the set up for zero-copy?
        auto publisher = create_publisher<CardMsg>("/hand", 10);
        if (publisher.can_loan_messages()) {
            auto loaned = publisher.borrow_loaned_message();
            // fill in‐place
            loaned.get() = to_msg(card);
            // zero‐copy handoff
            publisher.publish(std::move(loaned));
        } else {
            publisher.publish(to_msg(card));
        }
    }

    void ComposableFiveCardStudDealer::on_show_hand(const ShowHand::SharedPtr msg) {
        // Business Logic Here
    }
```

And in player:

```cpp
class ComposableFiveCardStudPlayer : public rclcpp::Node {
  public:
    ComposableFiveCardStudPlayer(
        rclcpp::NodeOptions options) : Node("composable_five_card_stud_player", options.use_intra_process_comms(true)) {
        rclcpp::SubscriptionOptions sub_opts;
        sub_opts.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

        // Zero copy messaging with unique ptr to a CardMsg
        card_sub_ = this->create_subscription<CardMsg>(
            "/hand",
            rclcpp::QoS{10},
            [this](CardMsg::UniquePtr msg)   // <- unique_ptr, no extra copy
            {
                std::lock_guard lk(mutex_);
                card_buffer_.push_back(five_card_stud::from_msg(*msg));
                if (card_buffer_.size() >= CARD_NUM) {
                    cv_.notify_one();
                }
                RCLCPP_INFO(
                    this->get_logger(),
                    "Received card: %s", card_buffer_.back().str().c_str());
            },
            sub_opts);
    }
};
```

On an intra-process topic, the publisher holds onto a single shared pointer until all intra-process subscribers consume it. Now the dealer and all three players live in the same component container and you turned on use_intra_process_comms(true) for every node. That replaces normal DDS delivery with the zero-copy “in-process” path.

### 1. Missing Messages

Now, why do we want a mix of inter/intra-process-comm topics? This is because:

- Intra-process transport hands you a single shared pointer and skips serialization, but it never puts the message into the DDS queues. The messages are stored in publisher. If the publisher (or subscriber) is torn down before your callback runs, that lone shared pointer gets destroyed—and your message vanishes. Turning IPC off forces every message through DDS, where your reliability, history, and durability QoS actually buffer and replay data across process boundaries.

- Many QoS features (e.g. transient_local durability, automatic retransmits on reliable) only live in the DDS layer. If you leave IPC enabled, you’ll silently lose those guarantees—even though you thought you’d set your publisher to be transient_local or “reliable.”

Generally, `MultiThreadedExecutor` plays well with intra-process transport:
    - Zero-copy messaging happens before the `MultiThreadedExecutor` executor touches message
    - `MultiThreadedExecutor` uses the same `rcl_wait_set_t` as the single-thread flavour, it just unblocks several worker threads to service ready entities concurrently. The intra-process manager still wakes the wait-set via a guard-condition exactly as in the single-thread case, so the transport layer is agnostic to the executor type. [See section "Receiving intra-process messages" in ROS2 intra process communication design doc](https://design.ros2.org/articles/intraprocess_communications.html?utm_source=chatgpt.com)

**So a rule of thumb is: disable IPC on topic publishing / subscribing if they actually involve inter_process_comm, or need to be delivered at destruction**.

### 2. No inter-topic ordering guarantee

This is a generic topic ordering issue that also applies to DDS. Each topic has its own queue; a later message on Topic B may arrive before an earlier message on Topic A. You cannot assume cross-topic arrival order:

```
_on_show_hand()  → stores cards  
_on_winner()     → _clear_game_state()  ← removes
```

- If callbacks run on different threads, you also lose control over which fires first.
- In DDS, **discovery can take up to 30 ms**—any messages published before discovery completes will be dropped.

### 3. “Latched” (transient-local) Topics Does Not Play Well With IPC

[See section "Incomplete Quality of Service support" in ROS2 intra process communication design doc](https://design.ros2.org/articles/intraprocess_communications.html?utm_source=chatgpt.com)

```cpp
winner_pub_ = this->create_publisher<std_msgs::msg::String>(
    "winner",
    rclcpp::QoS(1).reliable().transient_local());   //  ⟵ latch + reliable
```

- `transient_local()`: keeps the last sample alive for late-joining subscribers (“latched” behavior).
- `reliable()`: adds retransmits on loss (minimal overhead for small messages).
- Why disable IPC? The intra-process path only supports volatile durability. To combine `transient_local()` with zero-copy, you must turn off intra-process comms on that publisher.

    ```cpp
    rclcpp::SubscriptionOptions opts;
    sub_opts.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
    winner_pub_ = this->create_publisher<std_msgs::msg::String>(
        "winner",
        rclcpp::QoS(1).reliable().transient_local(),
        opts
    );
    ```

## Exception Catching

## When an Exception is Thrown in a ROS 2 Node

### During Construction

1. Your node’s constructor throws (e.g., `throw std::runtime_error(...)`).
2. `pluginlib` catches the low-level `class_loader::CreateClassException` and **re-throws** it as a `pluginlib::CreateClassException`.
3. `ComponentManager` has a `try…catch` around `createUniqueInstance()` and **catches the first exception**, beginning teardown.
4. During cleanup, `pluginlib`'s destructors for half-built objects detect the stored `std::exception_ptr` and call `std::rethrow_exception(eptr)`.
5. No enclosing `catch` in `ComponentManager` handles this second throw → the C++ runtime calls `std::terminate()`.
6. The process aborts

### During Runtime

1. A callback (e.g., timer, subscription, service handler) throws an exception.
2. The exception **propagates out of the callback** into the executor’s `spin()` loop.
3. `rclcpp` does **not catch** the exception by default → it goes unhandled.
4. The C++ runtime invokes `std::terminate()` on the uncaught throw.
5. The process aborts (typically printing the exception type, then receiving SIGABRT).
6. **All nodes in the container die**, just as in the construction failure case.
