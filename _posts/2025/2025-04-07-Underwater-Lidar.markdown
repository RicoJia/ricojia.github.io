---
layout: post
title: "Underwater LiDAR: How Light-Based Sensing Works Beneath the Waves"
date: 2025-04-07
categories: [robotics, sensors]
tags: [lidar, underwater, sensing, optics]
---

## Introduction

LiDAR (Light Detection and Ranging) is incredibly effective in air, enabling autonomous vehicles, drones, and robots to map their surroundings with centimeter-level precision. But what happens when you take LiDAR underwater? The answer: **it works, but very differently**.

Underwater LiDAR systems only work well over **short distances (meters, not kilometers)**. Water is _not_ transparent in the LiDAR sense, and two key phenomena explain why.

---

## 🔬 1. Why water limits LiDAR

### (1) Absorption

Water **absorbs light energy** at different rates depending on wavelength.

- Red light disappears quickly (~first few meters)
    
- Blue/green penetrates best → that’s why oceans look blue

### (2) Scattering (the real killer)

Particles (sediment, plankton, bubbles) cause light to bounce everywhere.

Instead of:

- clean reflection → you get
    
- **a fog-like glow (backscatter)**

This is similar to:

- Headlights in fog
- Shining a flashlight in dusty air

---

## 📏 2. Range comparison

|Environment|Typical LiDAR range|
|---|---|
|Air|100–1000+ meters|
|Clear water|~10–30 meters|
|Murky water|**< 5 meters (sometimes <1 m)**|

---

## 🎯 3. Real-world applications

Despite the limitations, underwater LiDAR systems are used in:

- Autonomous underwater vehicles (AUVs)
- Mine detection / naval mapping
- Archaeology (shipwreck mapping)
- Short-range inspection (pipes, structures)

---

## 🧠 4. How underwater LiDAR still works

Engineers adapt in a few clever ways:

### ✅ Use blue-green lasers (~450–550 nm)

This wavelength travels **farthest in water** because:

- **Water's absorption spectrum**: Water molecules absorb red and infrared light very quickly (within a few meters), but blue-green wavelengths have the **lowest absorption coefficient** in pure water
- **Selective scattering**: While scattering still occurs, blue-green light experiences less total attenuation than other wavelengths
- **Natural phenomenon**: This is the same reason oceans look blue—blue-green light penetrates deepest and gets scattered back to our eyes

The ~450–550 nm range represents the "optical window" in water, similar to how certain radio frequencies work best through the atmosphere.

### ✅ Time-gated detection

Ignore early scattered photons and only accept photons that arrive at the "correct" time. This filters out backscatter noise.

### ✅ High-power pulses

Use stronger signals to overcome absorption and scattering losses.

### ✅ Close-range operation

Most systems are designed for **inspection, not long-range mapping**, operating within meters rather than hundreds of meters.

---

## 🤔 5. What happens in murky water?

In really murky water:

- Light scatters so much that the beam turns into a glowing cloud
- Returns become noisy or meaningless

👉 At that point, LiDAR basically **fails**.

---

## 🔄 6. What's used instead?

When water gets bad, systems switch to:

### 🔊 SONAR (sound-based)

- Works great in murky water
- Long range (10s–1000s of meters)
- Lower resolution than LiDAR

---

## 📸 7. How underwater LiDAR generates data

Underwater LiDAR systems create 3D point clouds and images through the following process:

1. **Pulse emission**: A blue-green laser fires short pulses toward the target
2. **Time-of-flight measurement**: The system measures how long it takes for reflected photons to return
3. **Distance calculation**: Distance = (speed of light in water × time) / 2
4. **Scanning pattern**: The laser beam sweeps across the scene (using rotating mirrors or scanning mechanisms)
5. **Point cloud generation**: Each measurement creates a 3D point (x, y, z coordinates)

The "raw image" is essentially a collection of these time-stamped photon detections, which are then processed to filter out noise, correct for water properties, and generate a clean 3D representation of the underwater environment.

---

## 🎯 Summary

- **LiDAR works underwater**, but only for short ranges (meters, not kilometers)
- **Blue-green lasers** (~450–550 nm) penetrate water best due to minimal absorption
- **Scattering** from particles is the main challenge, especially in murky water
- **SONAR** is the go-to alternative when water clarity is poor
- Applications focus on **close-range inspection** rather than long-range mapping
