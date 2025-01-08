---
layout: post
title: Robotics Fundamentals - GPS and Its Coordinate System
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

## RTK GPS

RTK (Realtime-Kinematics) GPS needs an extra ground station that sends correction data to itself. The general workflow is:

1. A GPS Ground stateion is a known location. Its location is determined during a survey-in
1. The GPS ground station gets its GPS reading from the 4 satelelltes through trillateration.
1. Using its known location, the GPS ground station calculates the impacting factors such as the Atmospheric Delay, etc.
1. An RTK-GPS module is a.k.a a rover. It connects to a nearby GPS ground station. The baseline distance is typically 10-20km.
1. The GPS ground station start to send correction data that contains the impacting factors to the RTK-GPS data real time.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/8025ac22-946d-4663-86c9-6c8a335d49bf" height="300" alt=""/>
        <figcaption><a href="https://geo-matching.com/articles/which-is-better-among-static-survey-rtk-or-ppk">Source</a></figcaption>
    </figure>
</p>
</div>

Another crucial technique RTK GPS uses is Carrier Phase Ambiguity resolution. The carrier wave is a high frequency sinusoidal wave. In the illustration below, we can determine the distance between a satellite and a rover using the number of phases. Here, the total number of wavelength is $\phi = \alpha + \beta + N$. We aim to solve for N:

- `N`: an integer that represents a **fixed number of wavelengths**. This number is unknown at the moment
  - The L1 carrier wave is transmitted at 1575.42 MHz, its wavelength is 19cm.
- $\alpha$ is the **fractional number of wavelengths**. E.g., if the total distance is in total 10.5 wavelengths. $\alpha = 0.5$
- $\beta$ is the "accumulated number of observed wavelengths during the measurement period." As the receiver moves and the satellite orbits the earth, $\beta$ shrinks when the two gets closer, and expands when the two gets farther apart.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/c761fa90-d3f4-4d38-a2aa-1ba21b0241c6" height="300" alt=""/>
        <figcaption><a href="https://www.calian.com/advanced-technologies/gnss/information-support/gnss-positioning-techniques/">Source</a></figcaption>
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

## RTK GPS in Autonomous Vehicles

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
