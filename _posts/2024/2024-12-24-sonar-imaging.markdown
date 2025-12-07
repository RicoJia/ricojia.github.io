---
layout: post
title: Robotics - A Short Introduction To Underwater Sonar Imaging Cameras
date: '2024-12-24 13:19'
subtitle: Beam Forming
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - Oil and Gas
catalog: true
---

```
--> Gain (Rx/Time Varying Gain) --> Detection
```

## Beam Forming

## Amplification

After we’ve formed the beams, we wait for echoes to arrive and then run them through the receive chain.

Rx gain is applied in the analog front end at the hydrophone’s amplifier. It boosts all received signals before digitization so they sit in a usable range for the ADC and later processing, instead of being buried in quantization noise.

TVG (Time Varying Gain) is a range-dependent gain applied as a function of time. As sound travels farther, its energy spreads out and is absorbed by the water, so distant echoes are much weaker than near ones. TVG compensates for this by applying more gain to later (farther) samples and less gain to earlier (closer) samples. The result is that targets at different ranges become more comparable in amplitude, making a single detection threshold practical across the whole water column.

## Detection and Synchronization

With beamforming and amplification done, we can finally decide whether and where each beam “sees” a target.

For each beam at angle θ, we have a time series E(θ,t) representing the signal strength (often expressed in dB) as a function of time `t`, which maps to range. Based on this time series, we can choose different detection strategies:

- FAT (first above threshold): detect the first return that's above a certain noise threshold
- MAX: detect the strongest return regardless of when it arrives

Aligning pings and range calculation depends heavily on timing. For synchronization, we use Chrony, a highly accurate Network Time Protocol.
