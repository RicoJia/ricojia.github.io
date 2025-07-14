---
layout: post
title: Robotics Fundamentals - [ESKF Series 2] GPS and Its Coordinate System
date: '2024-03-10 13:19'
subtitle: RTK GPS, UTM Coordinates
comments: true
tags:
    - Robotics
---

There are two types of GNSS:

- Traditional single-point GPS: 10m accuracy, usually found in phones
- RTK (Real-Time-Kinematics) GPS. centimeter accuracy. It talks to a base station. Each module gives its `(x, y)`.

## GPS

A GPS needs to talk to at least 4 satellites to run a trillateration process to determine its current pose. Process: TODO

A GPS module's performance is affected by:

- Atmospheric delay: signals can be delayed as they pass through the ionosphere and troposphere.
- Multipath errors: signals can bounce off surfaces like buildings or mountains
- Satellite geometry: misalignment of satellite can reduce accuracy

A consumer grade GPS is 10m accuracy 95% of the time

0 deg meridian is a line that runs through Greenwich. 180 deg meridian is a line runs from North to the south pole. $It goes through Chukotka$

### How GPS Trilateration Works

Given:

- Four satellites with known positions $ \mathbf{S}_i = $x_i, y_i, z_i$ $ for $ i = 1, 2, 3, 4 $,
- Measured distances \$ r_i \$ from receiver to each satellite.
We want to find the unknown receiver position $ \mathbf{x} = $x, y, z$ $.

For each satellite, we have:
$$
\|\mathbf{x} - \mathbf{S}_i\|^2 = r_i^2.
$$

Subtract satellite 4’s equation from others:
$$
\|\mathbf{x} - \mathbf{S}_i\|^2 - \|\mathbf{x} - \mathbf{S}_4\|^2 = r_i^2 - r_4^2.
$$
Expanding and simplifying:

$$
2(x_4 - x_i)x + 2(y_4 - y_i)y + 2(z_4 - z_i)z
= r_i^2 - r_4^2 - \left[(x_i^2 - x_4^2) + (y_i^2 - y_4^2) + (z_i^2 - z_4^2)\right].
$$

This yields a linear system:
$$
A \mathbf{x} = \mathbf{b},
$$

where:

- $ A \in \mathbb{R}^{3 \times 3} $ has rows $ 2(\mathbf{S}_4 - \mathbf{S}_i)$,
- $ \mathbf{b} \in \mathbb{R}^3 $ is the simplified right-hand side.

If overdetermined or noisy, solve using the pseudoinverse:
$$
\hat{\mathbf{x}} = A^\dagger \mathbf{b}.
$$

This gives the least-squares estimate for the receiver's position.

## RTK GPS

RTK (Realtime-Kinematics) GPS needs an extra ground station that sends correction data to itself. The general workflow is:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/Pqh0mvgt/400820171-8025ac22-946d-4663-86c9-6c8a335d49bf.jpg" height="300" alt=""/>
    </figure>
</p>
</div>

1. A GPS Ground stateion is a known location. Its location is determined during a survey-in
2. The GPS ground station gets its GPS reading from the 4 satelelltes through trillateration. Each satellite starts broadcasting a message that's received by the base station and rovers.
3. The base station broadcasts a satellite message. A nearby RTK-GPS rover (<10km>) hears it, and calculate correction factors
    - It's called RTK because correction factors are calculated as the model moves ("kinematics")

### Correction Factors

#### Modelling

$$
\begin{gather*}
\begin{aligned}
p = r + c(\delta t_u - \delta t^{(s)}) + I + T + \varepsilon_p \\
\varphi = \lambda^{-1} \left[ r + c(\delta t_u - \delta t^{(s)}) - I + T \right] - N + \varepsilon_\varphi
\end{aligned}
\end{gather*}
$$

- $p$: Pseudorange observation (meters).It's the `c * apparent_time_difference`
  - So the formula basically says "the pseudorange is the sum of the true satellite-receiver distance, clock bias (difference), and various delays"
- $r$: True geometric distance between satellite and receiver (m)
- $c$: Speed of light (m/s)
- $\delta t_u$: Receiver clock bias (s)
- $\delta t^{(s)}$: Satellite clock bias (s)
- $I$: Ionospheric delay (m)
- $T$: Tropospheric delay (m)
- $\varphi$: Carrier-phase observation (in cycles or radians)
- $\lambda$: Carrier wavelength (m)
- $N$: Integer ambiguity (in cycles)
- $\varepsilon_p$, $\varepsilon_\varphi$: Measurement noise or unmodeled error

 The observation equations include aggregated error terms $\varepsilon$, representing unmodeled effects. For simplicity, these terms are omitted in subsequent formulas.

In step 2, ionospheric effect could cause delays experienced by receives and base stations. If a rover is close to a base station, the ionospheric delay is roughly the same:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/C5QV7Mjc/Screenshot-from-2025-07-13-17-11-50.png" height="300" alt=""/>
    </figure>
</p>
</div>

#### Single-, Double-, and Triple-Difference Observation Equations

The base station sees:
    - Apparent time (a.k.a pseudo range) - $t_{true} + t_{delay}$
    - Phase: (see below)

At the rover side, by taking the difference between the satellite message it hears directly and that from the base station, the resulting single-difference observation is:

$$
\begin{gather*}
\begin{aligned}
p_{ij}^p = r_{ij}^p + c \delta t_{u,ij}^p \\
\varphi_{ij}^p = \lambda^{-1} \left( r_{ij}^p + c \delta t_{u,ij}^p \right) - N_{ij}^p
\end{aligned}
\end{gather*}
$$

- Subscripts $i$, $j$: Receiver indices for the rover and satellite
- $p_{ij}^p$: Single-differenced pseudorange
- $\varphi_{ij}^p$: Single-differenced carrier-phase
- $r_{ij}^p$: Single-differenced geometric distance
- $\delta t_{u,ij}^p$: Single-differenced receiver clock error
- $N_{ij}^p$: Single-differenced integer ambiguity

[More details can be seen here](https://blog.csdn.net/qq_41782151/article/details/118601308?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-118601308-blog-81043370.pc_relevant_multi_platform_whitelistv4&spm=1001.2101.3001.4242.1&utm_relevant_index=3)

### Carrier Phase and Wavelength Counts

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/c4Rp6czD/400820171-8025ac22-946d-4663-86c9-6c8a335d49bf.jpg" height="300" alt=""/>
        <figcaption><a href="https://www.calian.com/advanced-technologies/gnss/information-support/gnss-positioning-techniques/">Source</a></figcaption>
    </figure>
</p>
</div>

Another crucial technique RTK GPS uses is Carrier Phase Ambiguity resolution. The carrier wave is a high frequency sinusoidal wave. In the illustration below, we can determine the distance between a satellite and a rover using the number of phases. Here, the total number of wavelength is $\phi = \alpha + \beta + N$. We aim to solve for N:

- `N`: an integer that represents a **fixed number of wavelengths**. This number is unknown at the moment
  - The L1 carrier wave is transmitted at 1575.42 MHz, its wavelength is 19cm.
- $\alpha$ is the **fractional number of wavelengths**. E.g., if the total distance is in total 10.5 wavelengths. $\alpha = 0.5$
- $\beta$ is the "accumulated number of observed wavelengths during the measurement period." As the receiver moves and the satellite orbits the earth, $\beta$ shrinks when the two gets closer, and expands when the two gets farther apart.

### RTK GPS in Autonomous Vehicles

In an autonomous vehicle, we use two RTK GPS modules (蘑菇头) so we know the mid point `(x, y, z)`, and its heading $\theta$. This set up is also known as DUAL GNSS Compassing. **Usually, GPS signals come in at 1hz.**

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/009c1827-c12c-410f-a0eb-20b8c051a312" height="300" alt=""/>
        <figcaption><a href="">RTK GPS </a></figcaption>
    </figure>
</p>
</div>

There are three ways to mount the RTK. The angle between the line and the heading of the car is an "extrinsic rotation":

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/b5ac7d22-e6e0-4b97-9cdb-f6bb9e496330" height="300" alt=""/>
    </figure>
</p>
</div>

## Longitude - Latitude System

Longitude and latitude is a very well-known concept. Some drawbacks are:

- Requires many significant digits to represent a location precisely
- Not directly in meters
- The polar areas have a **singularity**, because longitude `0 - 180 deg` converges there. So any longitude value
could be assigned to the north pole, but in the meantime, large changes in the longitude would happen, too.

## UTM Coordinates

The Universal Transverse Mercator (UTM) coordinate system can better handle the significant digits issue better. Singularity issue? Its workflow is:

- Projects the globe onto a map
- Segment the map into 60 parts (longitude) and 20 parts (latitude).
  - So each zone is 6 deg longitudinally, 8 deg latitudinally.
  - The 20 latitudes are represented by a letter in `C to X`, exluding `I` and `O`
    - Zone `33T` represents zone `33` in the T latitudinal region
- Each zone has a coordinate frame `x = left = east,  y=up=norh`
- The center line is x = 500km
- UTM has opposite north directions in the southern hemisphere

Advantages:

- Directly in meters!
- Fewer sigificant digits are required
  - `500km` is a common scale requires 6 significant digits. So to represent centimeters, one needs 8 significant digits (FP64)

Drawbacks include:

- Region number are required in representation
  - So cross-zone operations should be handled separately. But some RTK systems help us handle this.
- Distortions are large in the polar zones

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/f56f692b-bedc-445a-8c59-2790d489e0a3" height="300" alt=""/>
    </figure>
</p>
</div>
