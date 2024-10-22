---
layout: post
title: Robotics - 2024 3D Lidar Selection For Robotics
date: '2024-10-22 13:19'
subtitle: Livox, Unitree Lidars
header-img: "img/post-bg-unix"
tags:
    - Robotics
comments: true
---

This article is inspired by [this YouTube video made by 大刘科普](https://www.youtube.com/watch?v=ceWjU4A3yG4)

## Brief Overview Of How 3D LiDAR (Light Detection and Ranging) works

There are 2 types of LiDARs: ToF LiDARs and FMCW LiDARs. The mainstream is ToF, which has 3 types: mechanical (机械式激光雷达), hybrid sold-state (混合固态激光雷达), and fully solid-state (全固态激光雷达). This classification is based on if there are moving parts in the LiDAR.

### Concepts

#### Multi-Channel LiDAR

Multi-Channel LiDARs such as a 16, 32, or 64-channel LiDAR have multiple channels, or emitter-receiver pairs stacked vertically. Their vertical fov is typically ±10° to ±30°. Each receiver receives light from a fixed angle. The angle is determined by the orientation of the emitter and the receiver pair.

Imagine a stack of lasers and detectors aligned vertically, each pointing at a slightly different angle up or down. As the LiDAR spins around its axis, these channels sweep out horizontal slices at their respective vertical angles, collectively scanning the surrounding environment in 3D.

Multi-channel LiDARs are widely used in self-driving cars for obstacle detection and environment mapping. ToF cameras (Flash LiDAR) are used more in consumer electronics.

LiDAR systems are engineered to prevent interference between channels.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/af5d77fa-8bb9-4f6c-ab31-851e20531ff9" height="300" alt=""/>
        <figcaption><a href="https://www.nature.com/articles/s41598-022-26394-6">Source: Nature</a></figcaption>
    </figure>
</p>
</div>

#### Single Return vs Dual Return

The term "dual return" refers to the system's ability to detect and record multiple reflections of a single laser pulse as it encounters objects at different distances in its path. This is particularly useful in environments where the laser pulse may pass through partially transparent or semi-obstructive objects like vegetation, windows, or dust, and then reflect off a more solid object behind them.

- In single return mode, the LiDAR system records only the first reflection (the closest object) for each laser pulse.
- In dual return mode, the LiDAR system records two reflections for each laser pulse.

### ToF (Time of Flight) LiDAR Working Principle

1. Emission: the LiDAR emits short, intense pulses of laser light, typically in the near-infrared spectrum.
2. Direction: these pulses (beams) are directed to mirrors or lenses to achieve intended angles
3. Propagation: the laser pulses travel through the air at the speed of light
4. Reflection: when the pulses encounter an object, a portion of the light is reflected back towards the LiDAR sensor.

$\Delta t$ is measured between the emission and reception of the laser pulse with picosecond ($10^-{12}$s) resolution. The distance is $d = \frac{c \times \Delta t}{2}$

To generate a 3D depth map, **a ToF LiDAR's receive can be modeled as a pinhole camera model, where each pixel on the field of view corresponds to a point that's at an angle to the optical center.**

<p align="center">
<img src="https://github.com/RicoJia/The-Dream-Robot/assets/39393023/aa1eb110-f272-4939-b586-44eecae787ef" height="300"/>
</p>

### Mechanical LiDAR

There is a step motor that drives a vertical lidar emitter to scan the environment. The mechanical part is hard to maintain. Example: Google car?

### Hybrid Solid State LiDAR

#### MEMS mirrors (微振镜)

There is a mirror that rotates driven by MEMS.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/445bc30a-3b71-43f1-912d-2644c5c5b63f" height="300" alt=""/>
    </figure>
</p>
</div>

#### Polygon Mirror (转镜)

There is a 2D mirror array that actually rotates to reflect the beam. It's the first LiDAR type that passed the automobile grade verification (2017, Audi A8)

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/544ae7e8-c514-49ac-90ae-d088fe6b1fd1" height="300" alt=""/>
    </figure>
</p>
</div>

So in 2D, multiple beams (channels) can be emitted.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/4bd599e9-34c4-4f26-a50d-e0a14be6b525" height="300" alt=""/>
    </figure>
</p>
</div>

#### Prism 棱镜 (adopted by DJI Livox, used on XPeng P5)

There are two prisms that are rotating at **different speeds**. The laser beam goes through them, and because of different refractions at different parts of the prisms, the laser beam will be refracted to different 2D angles.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/c77d11a0-adcf-49eb-928d-ce55ce49afab" height="300" alt=""/>
    </figure>
</p>
</div>

Unlike the other types whose laser patterns are grids, the laser pattern of the prism LiDAR looks like a daisy

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/4afc8756-8881-4dbc-a23e-e6c3e38a4a3f" height="300" alt=""/>
    </figure>
</p>
</div>

## Solid-State LiDAR

### OPA (Optical Phased Array, 相控阵)

When light travels through an aperture, This is called "diffraction", different parts of the beam gets different phases.

Laser diffraction pattern when projecting onto a plate after travelling through an aperture.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/235eeef8-6fdd-4087-90c6-e58f9d3ae41a" height="300" alt=""/>
    </figure>
</p>
</div>

In OPA, a laser pulse goes through an array of apertures. The phase of light beams are controlled by the diameter of the individual apertures. Then, constructive and destructive interference of lights from direfferent directions will change the direction of the direction of the net light. A typical OPA LiDAR can achieve range accuracies on the order of centimeters.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="TODO" height="300" alt=""/>
    </figure>
</p>
</div>

### Flash LiDAR, or ToF Cameras (iPhone, Ouster)

Flash LiDAR emits a 2D plane of lights at once (like a camera flash), then it records the time of flight of the beams reflected on to the receiver pixels. It has a relatively wide field of view but lower resolution. So it is always used on consumer electronics but not used on cars. So Flash LiDARs can be thought of as RGBD cameras without RGB.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/77af1ab8-0b5a-4401-a737-6b2d256e4c1a" height="300" alt=""/>
    </figure>
</p>
</div>

| Feature                    | OPA LiDAR                                         | Flash LiDAR                                      |
|----------------------------|--------------------------------------------------|--------------------------------------------------|
| Range Accuracy              | Centimeter-level accuracy at longer ranges       | Centimeter-level accuracy at shorter ranges      |
| Angular Resolution          | High (dependent on array size and element spacing) | Moderate (limited by detector pixel size)        |
| Field of View               | Moderate (limited by steering range)             | Wide (up to 120° or more)                        |
| Update Rate                 | High (electronic beam steering)                  | Very High (captures entire scene instantly)      |
| Effective Range             | Up to several hundred meters                     | Typically up to 50 meters                        |
| Suitability for Cars        | Ideal for automotive applications needing long-range detection | Less suitable due to range limitations           |

### Disadvantages of ToF Based LiDARs

- Sunlight: a major source of noise for outdoor LiDARs
- Thermal noise: a major source of noise that's caused by the random fluctuations in the electrical signals, due to the temperature-dependent motion of charge carriers in the detector or receiver electronics.
- Dark Current: Dark current is a constant current that flows through a photodiode even when no light is present. It's a thermal phenomenon caused by electrons spontaneously generated within the silicon chip.
- Interference between LiDAR laser signals.

## Important Area of Development - Increase Wavelength for Safety

LiDAR's wavelength cannot get below 850nm. The mainstream choice in the indsutry is 905nm. This value is near-infrared but is close to visible light. So it can potentially damage the retina if input power is too high. So, nowadays, manufacturers are looking for ways to reduce the wave length down to 1550nm. 1550nm waves can be absorbed by liquids in the vitreous body. So 1550 nm LiDAR can have a higher output power and increase detection performance.

However, 1550nm LiDAR is still expensive because its emitter and receiver CMOS come from optical communication manufacturers (1550nm is also used there). Its emitter uses InP (磷化铟), and its receiver uses InGaAs (铟镓砷)

Meanwhile, 905nm laser can be received by silicon CMOS.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/badf05cc-5b9f-4162-80bc-d49940bf8534" height="300" alt=""/>
    </figure>
</p>
</div>

## FMCW (Frequency-Modulated Continuous Wave)

FMCW LiDAR emits a continuous wave that is frequency-modulated over time (e.g., a linear chirp). The frequency of the wave increases or decreases in a known pattern, typically in a sawtooth or triangular waveform. The emitted wave reflects off an object and returns to the LiDAR sensor. The received wave is then mixed with a portion of the original emitted wave (the "reference wave").

The frequency difference of waves when emitted and received is the beat frequency. This beat frequency is directly related to the distance to the object (since it corresponds to the time delay) and the velocity. This is called the **Doppler Effect**.

Beat freuncy is determined by:
$$
f_b = 2 \frac{d}{c} S + 2 \frac{v}{\lambda}
$$

- d is the distance to the target,
- v is the relative velocity,
- c is the speed of light,
- S is the slope of the frequency modulation (Hz/s),
- λ is the wavelength of the laser.

To calculate beat frequency, the received saved is mixed with the emitted wave, this is called "coherent detection". Mixing here involves multiplying two signals and results in frequencies that are the sums and differences of the two waves. So to get the beat frequency (the difference), one filters out the high frequency components and apply Fast Fourier Transform (FFT).

To solve for distance `d` and velocity `b`, FMCW LiDAR increases and decreases the emitter frequency, which is also called "up-chirp" and "down-chirp". So beat frequencies become:

$$
f_{b, up} = S_{\tau} + f_D
\\
f_{b, down} = -S_{\tau} + f_D
$$

### Advantages of FMCW

In general, FMCW has a higher signal-noise ratio (SNR) because:

- the noise typically lacks the specific frequency and phase characteristics of the transmitted signal, it doesn't contribute significantly after the coherent detection process
- The resulting beat frequency (difference frequency) is much lower than the optical carrier frequency, so narrow-band filters can be applied.

Another advantage of FMCW is as long as it hits an obstacle, from velocity, the car can quickly decides if there is a moving obstacle.

### Disadvantages of FMCW

- FMCW needs time to measure Doppler effect. ToF typically takes 2ms to process a batch of data, but FMCW could take as long as 20ms.
- FMCW is **expensive**: It's not mass-industrialized yet. In September 2024, Mobileeye shuts down its FMCW LiDAR development ☹️

## Industry Landscape

| Type                     | Advantages                          | Disadvantages                                    | Representative Companies               |
|--------------------------|--------------------------------------|-------------------------------------------------|----------------------------------------|
| Mechanical LiDAR          | Mature technology                   | Complex structure, large size, expensive core components | US: Velodyne, China: Hesai Technology, RoboSense |
| Hybrid Solid-State LiDAR  | Low cost, small size, strong performance, reliable | Technology is relatively immature               | US: Luminar, Innoviz, Ibeo, China: Huawei, RoboSense, Sagitar Juchuang |
| Fully Solid-State LiDAR   | Fully solid-state, small size, low cost | Short detection distance, low angular resolution, difficult manufacturing process | US: Innoviz, Ouster, Quanergy, China: RoboSense, Hesai Technology, Sagitar Juchuang, Northern United |

### ADAS Lidars

In The ADAS field, the main players are Chinese and US companies

- Chinese companies: Hesai Technology (禾赛科技), RoboSense (速腾聚创), and LeiShen Intelligent Systems (镭神智能)

| Company                   | Product                | Specifications | Price |
|----------------------------|------------------------|----------------|-------|
| **Hesai Technologies**      | Pandar 40p             | 40-channel     | $3,000 - $4,000 |
| **RoboSense**               | Helios 16p         | Solid-State? 16-channel, 150m, 288kpts/s        | $3,000-$4,000   |
| **RoboSense**               | RS-LiDAR-16 miniature         | Mechanical, 16-channel, 150m, 320kpts/s        | $442   |
| **LeiShen Intelligent Systems** | LSLIDAR 16-Line Mechanical LiDAR   |  16-channel, mechanical      | $3,000.00   |

### Robotics

| Company                   | Product                | Specifications | Price |
|----------------------------|------------------------|----------------|-------|
| **Livox**      | Mid-360 | 10cm - 70m,  40-channel,9-27v, 6.5 W     | $838 |
| **Unitree**    | 3D LiDAR L1 360 | 5cm - 30m         | Solid-State? 16-channel, 150m, 21.6kpt/s        | $3,000-$4,000   |

- UniTree claimed that although their L1's point rate is only `21.6 kpt/s`, it's on par with the automotive-grade LiDARs for point cloud density PER VOLUME (kpts/m/s)
