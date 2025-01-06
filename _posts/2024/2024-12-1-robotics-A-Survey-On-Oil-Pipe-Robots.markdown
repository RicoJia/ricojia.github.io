---
layout: post
title: Robotics - A Survey On Oil Pipe Inspection Robotics (Revising)
date: '2024-12-1 13:19'
subtitle: 
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - Oil and Gas
comments: true
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

## Inspection with 3D LiDAR Examples

- [Construction inspection using quadruped](https://www.youtube.com/watch?v=Eyl6II_tB3k)
- [3D Factory Mapping using Livox Mid-360](https://www.bilibili.com/video/BV1zb411Z731/?spm_id_from=333.337.search-card.all.click&vd_source=ae0bfd67b026e62fbc37ca190dfd1839)
