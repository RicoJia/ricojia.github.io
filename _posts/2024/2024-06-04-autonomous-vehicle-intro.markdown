---
layout: post
title: Autonomous Vehicle Introduction
date: '2024-06-04 10:11'
subtitle: 
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Autonomous Vehicle
---

## General Project Flow

1. What the robot sees - Perception
2. localization + mapping
3. Navigation - Motion Planning
4. Planning + control

### Autonomy Levels from SAE (Society of Automotive Engineers)

| Level       | Level 0 (L0)       | Level 1 (L1)         | Level 2 (L2)         | Level 3 (L3)                  | Level 4 (L4)       | Level 5 (L5)            |
|-------------|--------------------|----------------------|----------------------|-------------------------------|--------------------|-------------------------|
| **Driver Role** | The driver is responsible for driving the vehicle | The driver is responsible for driving the vehicle | The driver is responsible for driving the vehicle | The computer is responsible for driving the vehicle | The computer is responsible for driving the vehicle | The computer is responsible for driving the vehicle |
| **Is Monitoring Needed?** | The driver must always be prepared to take over | The driver must always be prepared to take over | The driver must always be prepared to take over | The vehicle will request the driver to take over if needed | No need to monitor | No need to monitor |
| **Typical Functions** | AEB: Automatic Emergency Braking<br> BSD: Blind Spot Detection<br> LDW: Lane Departure Warning | ALC: Intelligent Lane Change Assist<br> LCC: Lane Centering Control<br> ACC: Adaptive Cruise Control | LCC + ACC | Traffic Jam Pilot<br> Automatic Parking<br> Autonomous Summoning | Robotaxi<br> Robotrack<br> Remove directional input and autonomous driving pedals | All-weather autonomous driving |

Humans and robots are different. This is similar to "should planes be similar to birds?"

As of 2024, most EVs in China have L2 auto-driving.

- ACC (adaptive cruise control),
- LCC (Lane Centering Control). ALC: (Automated Lane Change). These can be achieved by pure computer vision + short-distance ultrasonic solutions
- AEB (Automatic Emergency Braking)
- BSD (Blind Spot Detection)
- LDW (Lane Departure Warning)

There are two main differences between L2 and L4:

- L2 allows the human driver to take over, L4 does not.
- In most cases, L2 generally requires clear vision of lanes, and does not guarantee driving autonomy. L4 however, does.

## Current Market Situation (Circa 2021)

Most autonomous vehicle manufacturers are not being aggressive, because currently human drivers are still required. However, for delivery cars or cleaning robots, they need L4.

- Low speed L4 applications: mine trucks, delivery robots, cleaning robots
- High speed L4: Robotruck, robotaxi, robot bus

There's a metric "Miles Per Intervention" (MPI) that manufacturers. The lower, the better. In general, L4 Robotaxi companies have fewer vehicles than car manufacturers. Also, their stakes are higher if their technology fails.

| Company          | Number of Vehicles | Number of Disengagements | Miles Tested (Miles) | Average Miles Per Disengagement (MPI) |
|-------------------|--------------------|--------------------------|----------------------|---------------------------------------|
| Waymo            | 693                | 292                      | 2,325,843            | 7,965                                 |
| Cruise           | 138                | 21                       | 876,105              | 41,719                                |
| Pony.ai          | 38                 | 21                       | 305,617              | 14,553                                |
| Zoox             | 85                 | 21                       | 155,125              | 7,387                                 |
| Nuro             | 15                 | 23                       | 59,100               | 2,570                                 |
| Mercedes-Benz    | 17                 | 272                      | 58,613               | 215                                   |
| WeRide           | 14                 | 3                        | 57,966               | 19,322                                |
| AutoX            | 44                 | 1                        | 50,108               | 50,108                                |
| DiDi             | 12                 | 1                        | 40,745               | 40,745                                |
| Argo AI          | 13                 | 1                        | 36,734               | 36,734                                |
| Motional         | 2                  | 2                        | 30,872               | 15,436                                |
| Embark           | 6                  | 82                       | 28,004               | 342                                   |
| Toyota           | 4                  | 419                      | 13,959               | 33                                    |
| Apple            | 37                 | 663                      | 13,272               | 20                                    |
| Aurora           | 7                  | 9                        | 12,647               | 1,405                                 |
| Lyft             | 23                 | 23                       | 11,200               | 487                                   |
| Almotive         | 2                  | 106                      | 2,976                | 28                                    |
| Gatik AI         | 3                  | 6                        | 1,924                | 321                                   |
| NavInfo          | 3                  | 143                      | 1,635                | 11                                    |
| Baidu Apollo     | 5                  | 1                        | 1,468                | 1,468                                 |
| SF Motors        | 2                  | 61                       | 875                  | 14                                    |
| Nissan           | 5                  | 17                       | 508                  | 39                                    |
| FEV              | 2                  | 205                      | 336                  | 2                                     |
| EasyMile         | 1                  | 222                      | 320                  | 1                                     |
| Udelv            | 1                  | 46                       | 60                   | 1                                     |
| Jingchi Tech     | 2                  | 0                        | 39                   | -                                     |
| UATC             | 3                  | 31                       | 14                   | 0.5                                   |

For L2 driving, one does not need high precision maps. its perception is not exact

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/3c843171-1e66-4f6d-aee2-87abef12d502" height="300" alt=""/>
       </figure>
    </p>
</div>

For L4 driving, a high-precision map needs to be exact. This is also a burden.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/930d6e55-6e7e-43c2-afe3-dea9f35fd79b" height="300" alt=""/>
       </figure>
    </p>
</div>

High precision maps usually come from high-resolution satellite / drone / LiDAR maps.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/c1b4e535-ebd5-44fb-9c60-35f62bcef7a8" height="300" alt=""/>
       </figure>
    </p>
</div>
