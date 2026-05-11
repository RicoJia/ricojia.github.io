---
layout: post
title: Math - Integration Approximations
date: 2017-03-02 13:19
subtitle: Trapezoidal Rule
comments: true
tags:
  - Math
---
## Trapezoidal Rule

If the robot drives forward while turning left, using yaw_mid, moves the displacement along the middle of the arc rather than entirely along the old heading.

Integrate pose using a trapezoidal-style half-step yaw.  Plain Euler would project body-frame velocity using the yaw at the beginning of the timestep. If the robot is turning, the heading changes  during the interval, so we instead use the midpoint heading:  

Δx ≈ v dt cos(ψ + 0.5ωdt)
Δy ≈ v dt sin(ψ + 0.5ωdt)
Δψ = ωdt

This better approximates the curved arc traced by a robot that is translating while turning.



