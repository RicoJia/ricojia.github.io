---
layout: post
title: Robotics - A Survey On Oil Pipe Inspection Robotics (Revising)
date: '2024-12-20 13:19'
subtitle: Sonar Imaging
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - Oil and Gas
catalog: true
---

## Non-Destructive Testing Methods

American Society for Testing and Materials (ASTM) and American Society of Mechanical Engineers (ASME) develops standards for pipe inspections

- Ultrasonic Testing (UT)
  - Standard Practice for Ultrasonic Testing of Metal Pipe and Tubing ([ASTM E213-22](https://www.astm.org/e0213-22.html))
- Magnetic Particle Testing (MT)
  - Near-surface Notch Detection ([ASTM E709](https://www.astm.org/e0709-21.html))
- Radiographic Testing (RT)
- Visual Inspection (VT)
- Near-surface Eddy-Current-Testing ([ASTM E2884-22](https://www.astm.org/e2884-22.html))
  - [A quick tutorial on eddy current is here](../2017/2017-06-01-electronics-eddy-current.markdown)

Different liquids could cause different types of corrosions:

- `H2S` (Hydrogen-Sulfide) creates small punctures, whereas most other liquids create relatively larger corrosions.

## Major Players In The Market and Their Robot

[MPE Supplies](https://mpesupplies.com/products/dakotah-power-tools-500ft-robotic-crawler-inspection-system-sewer-crawler?gad_source=1&gclid=Cj0KCQiAr7C6BhDRARIsAOUKifgQB3tfUs9hEb4J3qpJmgopCde9cWLddTOfVa8txg_RJMf-WyWgh08aAoixEALw_wcB)

- Features: 360 deg flip, 220 deg rotate. Front light source, backlight source, video
- $20000
- Location: Arkansas

## Dilemma of ML in Pipe Inspection - Precision and Recall

Pipe inspection is labor-intensive. People don't trust ML that much. This lies in the ML's lower ["recall" and "precision" rates](../2022/2022-02-15-deep-learning-performance-metrics.markdown)

If the inspection agency missed a spot (false negative), it could bring lawsuits where we need to explain why the manual + AI system fails to detect that. In machine learning, this is characterized as **"recall"**.

If the system has too many false positives, then the system is too costly to verify. In machine learning, this is characterized as [precision](../2022/2022-02-15-deep-learning-performance-metrics.markdown) is low. In oil and gas, precision us termed as the "prove-up" rate:

$$
\begin{gather*}
\begin{aligned}
& p = \frac{\text{verified defects}}{\text{total detected defects}}
\end{aligned}
\end{gather*}
$$

Explaining why relatively low precision and recall happens is not easy, as ML systems are still largely "unpredictable" during run-time. What's worse is many customers and the court mistakenly think that ML systems should aim for perfect detection results, which is similar in the autonomous vehicle industry. In ML, we always compare the ML system with the "best human-level" performance.

### Gas Meter

- What is LEL in Gas meter? (Lower Explosive Limit)
  - Calibration: to set a baseline of "good air". But when you calibrate it in an H2S rich environment, it's not good
  - "Unknown gas 100" can be resolved by powercycling
  - Some gas meters have two modes: slow and fast
  - "Zero-air" mode means clean air. If a flammable gas is detected, it can trigger an alarm.

## Inspection with 3D LiDAR Examples

- [Construction inspection using quadruped](https://www.youtube.com/watch?v=Eyl6II_tB3k)
- [3D Factory Mapping using Livox Mid-360](https://www.bilibili.com/video/BV1zb411Z731/?spm_id_from=333.337.search-card.all.click&vd_source=ae0bfd67b026e62fbc37ca190dfd1839)

## Sonar Imaging

Sonar imaging cameras are able to create a 2D occupancy map or 3D point cloud.

A 2D sonar scannar sends a vertical planar sound wave around 360 degrees (horizontal). It also has 1 receiver that receives the signal. If there is an obstacle, we will get a ping. It's possible that at one fixed angle, we get multiple pings. That could be due to:

- Multiple objects being detected
- Multipath reflection taking place. This is the case where the sound wave gets reflected back to the receiver, but at a later time

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/5yq2NNZC/sonar-imaging.png" height="200" alt=""/>
        <figcaption>Sonar Imaging</figcaption>
    </figure>
</p>
</div>

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/1tR8nFrK/multipath.jpg" height="200" alt=""/>
        <figcaption>Multipath-Reflection</figcaption>
    </figure>
</p>
</div>

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/BQ2m48Mv/2d-sonar-scan.jpg" height="300" alt=""/>
        <figcaption>2D Sonar Scan</figcaption>
    </figure>
</p>
</div>

### How 3D Sonar Imaging Works


This is a very helpful document on learning about [the SeaBeam instruments Sonar Imaging Camera](https://lismap.uconn.edu/wp-content/uploads/sites/2333/2018/11/SeaBeamMultibeamTheoryOperation.pdf)

Below is a classic "Mills-cross" setup

1. Under the hull is a transducer array—hundreds-to-thousands of ceramic (piezo) elements laid out in a line (or sometimes two lines). The transducer creates a ping
    ![](https://i.postimg.cc/KYgp7CG9/Screenshot-from-2025-06-23-21-20-57.png)
2. Each ping goes a projector, which carefully applies delays (or phase shifts) to every element so the individual wavelets add up in a chosen direction and cancel elsewhere. (Constructive and Deconstructive Interference, Similar to Phase Array Radar). So this looks like a fan pattern

    ![](https://i.postimg.cc/SsFTgVjm/Screenshot-from-2025-06-23-21-21-03.png)

3. A receiver (hydrophone) array 
    1.  After the ping, each patch of the sea bed sends back a plane-wave (to very good approximation). Each receiver receives them and could form their own waveform: $A(t)$ (amplitude), and $\phi(t)$ (phase)
        - The receive array is at most a few metres long; the covered seabed distance (swath) is hundreds of metres away.
        - We assume the distance from the camera to sea bed is much longer than the receiver array size (far-field condition): `Range >> L^2/lambda`. A single echo from the seabed will have an arbitrary angle of incidence $\theta$ on to receiver 3, 2, 1. It will first reach 3, then travel by $d sin\theta$ to 2, then $d sin\theta$ to 1. 
            ![](https://i.postimg.cc/85Ry9WrM/Screenshot-from-2025-06-23-22-14-44.png)
        - **the spherical wavefront that leaves any single patch is practically planar by the time it spans the array.** Therefore, 
    2. So effectively, we can work out the **beams** to each patch of the seabed, based on the received waveforms at each receiver

        ![](https://i.postimg.cc/Kjm92znV/Screenshot-from-2025-06-23-22-14-25.png)

### Math for solving for distances: TODO


Leading players:

- Micron 
