---
layout: post
title: Robotics - [ROS2 Foundation - 1] DDS Notes
date: '2024-11-13 13:19'
subtitle: DDS, Zeroconf, IDL, FastRTPS, ROS2 Messaging Mechanism
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

----------------------------------------------------

## What's Middleware

Middleware abstracts the complexity of managing communication, serialization, discovery, and other low-level functionalities, allowing developers to focus on application-level logic.

- ROS1's middleware is TCPROS: it's a custom serialization format, a custom transport protocol as well as a custom central discovery mechanism

- ROS2 uses an existing middleware interface, DDS. There are multiple DDS implementations. ROS has a unified interface for these implementations
  - In DDS, each ROS 2 node is equivalent to a DDS "participant". There could be multiple ROS nodes in the same process. Each node is still a DDS participant
  - Under the hood, there are DDS datareader, datawriter, and DDS topics. They are not exposed to ROS users

```
+-----------------------------------------------+
|                   user land                   |   No middleware implementation specific code
+-----------------------------------------------+
|              ROS client library               |
+-----------------------------------------------+
|             middleware interface              |   No DDS specifics
+-----------------------------------------------+
| DDS adapter 1 | DDS adapter 2 | DDS adapter 3 |
+---------------+---------------+---------------+
|    RTI DDS impl 1 | PrimsTech DDS impl |    DDS impl 3 |
+---------------+---------------+---------------+
```

## Why DDS (Summary)

When trying to develop ROS 2 communication systems, Open Robotics faced two choices: 1. improving on the existing ROS1 infrastructure 2. Use ZeroMQ, Protocol Buffers, and Zeroconf (see below). Both choices need a middleware.

Some Salient Features of DDS:

1. The default implementation of DDS is **over UDP**, and only requires that level of functionality from the transport.
2. Because it's based on UDP, Quality of Service (QoS) needs to be introduced (unlike TCP)
    - for soft real-time, you can basically tune DDS to be just a UDP blaster.
3. DDS would completely replace the **ROS master** based discovery system.
4. DDS has publisher and subscriber. However, they are implemented by `DataWriter` and `DataReader`, which hides the implementations of communication
5. Shared Memory Support
    - In ROS 1, nodelets were used. Nodelets allow publishers and subscribers to share data by passing around `boost::shared_ptrs` to messages

Disadvantages of DDS:

- ROS must work within that existing design. If the design did not target a relevant use case or is not flexible, it might be necessary to work around the design.

### [1] DDS Implementations

There are multiple middlewares because you might have considerations such as license, platform availability, or computation footprint:

- `Fast DDS`(by eProsima) whose wire protocol is `Fast RTPS`. Package: `rmw_fastrtps_cpp`
  - ROS2 Humble's default
  - Check `echo $RMW_IMPLEMENTATION`.
- `Cyclone DDS` (Eclipse): slightly lighter weight, but less configurable than Fast RTPS. Can be explicitly installed by: `sudo apt install ros-humble-rmw-cyclonedds-cpp`. Package: `rmw_cyclonedds_cpp`
  - ROS2 Galactic's default
- `ConnextDDS` (RTI). Package: `rmw_connext_cpp`

DDS is controlled by:

```
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

- `rmw` stands for `ROS Middleware`. It' a package that implements the abstract ROS middleware interface using DDS or RTPS's API and tools.

Some RMW implementations can communicate:

- `Fast DDS <-> Cyclone DDS`
- `Fast DDS <-> Connext`

### [2] Zeroconf

ROS 2 and many other IoT devices heavily relies on Zeroconf. Zero Configuration Networking (`Zeroconf`) allow devices on local network to discover and communicate with each other without requiring manual configuation. It does:

- Automated Addressing:
  - Devices assign themselves a unique IP address without requiring a DHCP server. This is usually in the `link-local` address range `(169.254.x.x in IPv4)`
  - It tries to use DHCP first. If no DHCP server is found, the device assigns itself a link-local IP address.
- Service Discovery
  - Each device broadcasts and discovers services (printers, file shares, media servers) on the local network
  - Often uses **Multicast DNS**, **DNS Service Discovery (DNS-SD)**
- Name resolution
  - Based on the discovered devices, `names <-> IP` mapping can be broadcasted

Common protocols used inZeroconf :

- `Apple Bonjour`: A popular Zeroconf implementation used in macOS and iOS.
- `Avahi`: A Linux-based Zeroconf implementation, commonly used for service discovery.
- `DNS-SD (Service Discovery)`: Advertises and discovers services (e.g., "_http._tcp.local" for a web server).
- `Multicast DNS (mDNS)`: Resolves hostnames to IP addresses in the absence of a DNS server.

Advantages:

- No network configuration needed. It uses dynamic and flexible local networking
- Reduces dependency on network infrastructure like DHCP and DNS servers

Disadvantages:

- Scalability. In larger network, multicast could be noisy
- Security: open broadcasting could be intercepted
- Network scope: limited to a single subnet and does not cross routers

[Reference: ROS2 Design Documentation](https://design.ros2.org/articles/ros_on_dds.html)

----------------------------------------------------

## Backgrounds  of DDS

DDS comes out of a set of companies which are decades old, was laid out by the OMG which is an old-school software engineering organization, and is used largely by government and military users.

- They have limited forum support
- They do not have extensive user-contributed wikis or an active Github repository.

### [1] What is `Fast RTPS`

`Fast RTPS` was designed by Object Management Group (OMG). While DDS defines overall framework and API, `Fast RTPS` defines the nitty-gritty of the publisher-subscriber model used by `DDS`:

- Serialization, deserialization
- publishers, subscribers, topics
- QoS policies: reliability, durability, and liveliness
- Automatic Discovery
- Transport Independence uses UDP and shared_memory

### How does RTPS work?

First, RTPS has `Participants`, which represent applications or nodes in the distributed system. They correspond to DDS participants.

Each participant has `Writers` and `Readers`:

- Writers: Publish data (similar to DDS DataWriters).
- Readers: Subscribe to data (similar to DDS DataReaders).

Each writer and reader has `topics`: Logical channels through which writers and readers exchange data.

#### RTPS Communication Workflow

1. `discovery` process: RTPS uses Simple Endpoint Discovery Protocol (SEDP) to discover participants, topics, and endpoints (readers/writers).

2. `Data Exchange` phase:

- Writers publish serialized data to the network.
- Readers listen for data on the corresponding topic and deserialize it upon reception.

3. QoS Enforcement

### [2] What is IDL

DDS uses the “Interface Description Language (IDL)” as defined by the Object Management Group (OMG) for message definition and serialization. For example:

```
module RobotSystem {
  struct Pose {
    float x;       // Position in X-axis
    float y;       // Position in Y-axis
    float z;       // Position in Z-axis
    float roll;    // Rotation around X-axis
    float pitch;   // Rotation around Y-axis
    float yaw;     // Rotation around Z-axis
  };

  struct RobotState {
    string<255> name;  // Robot name (max 255 characters)
    Pose pose;         // Pose of the robot
    bool is_active;    // Is the robot active?
  };
};
```

DDS implementations (e.g., Fast DDS, Cyclone DDS, or RTI Connext) provide tools to generate code from the IDL file. The generated code includes:

    Serialization and deserialization functions.
    Definitions for the corresponding data types in the target programming language (e.g., C++, Python, etc.).

### What is ROS domain ID

ROS_DOMAIN_ID is an environment variable (default 0). It namespaces the DDS discovery ports so you can safely run multiple independent ROS 2 networks on the same physical LAN. ROS 2 uses it to compute DDS ports; pick 0–101 to avoid ephemeral-port conflicts

----------------------------------------------------

## Messages, Services and Actions

### [1] Message Conversions Base Mechanism

1. `.msg` files still keep its format and in-memory representation because it's proven to be functional in ROS1.
2. `.msg` file are converted into `.idl` files so that they could be used with the DDS transport.
    - Language specific files, conversion functions are generated as well.
        - Conversion functions convert ROS <-> DDS in memory instances
    - `ROS2` API only converts a message field-by-field into an `idl` object. DDS in-memory instances are not generated and consumed except only in `.publish()` and topic callbacks.
        - This is because serialization is at least `10x` longer than field-to-field copy.
        - ROS2 build can do `msg -> idl` outside `publish()` and receiver callbacks instead of doing a zero-copy msg conversion.
3. `.idl` files are published

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/a96902aa-42aa-4796-a121-01a071f97981" height="300" alt=""/>
    </figure>
</p>
</div>

So during c++ compilation, `msg` file `Pose.msg`:

```
float32 x
float32 y
float32 z
float32 roll
float32 pitch
float32 yaw
```

will be converted to:

```cpp
// Generated C++ code
namespace custom_msgs {
  struct Pose {
    float x;
    float y;
    float z;
    float roll;
    float pitch;
    float yaw;

    // Constructor
    Pose() : x(0.0), y(0.0), z(0.0), roll(0.0), pitch(0.0), yaw(0.0) {}

    // Comparison operators, serialization methods, etc., are also included
  };
}
```

So you can create:

```cpp
custom_msgs::Pose pose;
pose.x = 1.0;
...
```

In Python, similar thing happens during `build`:

```python
# Generated Python code
class Pose:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

# User code
from custom_msgs.msg import Pose

pose = Pose()
pose.x = 1.0
```

- In `ROS2` terms, each C++/Python variable is an **in-memory representation**
- The in-memory representation includes methods for serialization (converting to a binary format for transmission) and deserialization (reconstructing the object from the binary format).

### [2] Service and Actions

A service in ROS 2 is a mechanism for synchronous communication between a client and a server. It allows a node to send a request and receive a response. E.g,

```
# AddTwoInts.srv
int64 a
int64 b
---
int64 sum
```

An action is a mechanism for asynchronous, long-running tasks that provides periodic feedback and the ability to cancel the task. It uses `Goal-Feedback-Result` model:

```
# Fibonacci.action
int32 order  # Goal
---
int32[] sequence  # Feedback
---
int32[] sequence  # Result
```

Both Service and Actions are built on top of the `publish-subscribe` model.

- Service uses two topics: `/service_name_request` and `/service_name_response`
- Action has:
  - `Goal` service: send the goal to the action server
  - `Cancel` service: cancel the current goal
  - Feedback and status are communicated using topics:
    - A feedback topic for periodic updates.
    - A status topic to inform about the state of the goal (e.g., "active," "canceled," "succeeded").
  - `Result` service: retrieve the final result
