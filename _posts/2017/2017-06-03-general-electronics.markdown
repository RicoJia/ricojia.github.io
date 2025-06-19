---
layout: post
title: Electronics - General Electronics Notes
subtitle: Encoders, Motors, Capacitors, IMU, Testing, Telecommunication
date: '2017-06-03 13:19'
header-img: "img/bg-material.jpg"
tags:
    - Electronics
---
## Electronics

### EEPROM

EEPROM (Electrically Erasable Programmable Read-Only Memory) is a type of non-volatile memory used in microcontrollers to store small amounts of data that must be preserved when power is removed.

- Non-volatile (data remains in storage.)
- Writable and erasable (Unlike traditional ROM)

## Motors

### Quadrature Encoders

Quadrature means the design where two waves (square wave, sine wave, etc.) are 90 deg out of phase. In a quadrature encoder, two light sources are placed in slightly different positions, so their waves are 90 deg out of phase. With this design, we can tell the direction of rotation, and the number of ticks a motor has rotated.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/763f45b4-9b46-4306-a845-f574c6d9c8f7" height="300" alt=""/>
        <figcaption><a href="https://www.machinedesign.com/automation-iiot/article/21829959/review-of-quadrature-encoder-signals">Source</a></figcaption>
    </figure>
</p>
</div>

The two light sources produce a waveform:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/231fd3eb-8a02-427f-abcf-0ca145b874b9" height="300" alt=""/>
        <figcaption><a href="https://www.machinedesign.com/automation-iiot/article/21829959/review-of-quadrature-encoder-signals">Source </a></figcaption>
    </figure>
</p>
</div>

**One can see that if slots and hollows on the encoder disk are equal length, we can produce the clean square wave**. There are 4 combinations of pulses:

- Rising edge of Channel A.
- Falling edge of Channel A.
- Rising edge of Channel B.
- Falling edge of Channel B.

By looking at edge rising / falling status and the other channel's value, we can achieve "resolution = 4*360 deg / num_of_lines". This is how it works:

```python
def encoder_callback(channel):
    """Interrupt callback to handle encoder state changes."""
    global position, last_state_a, last_state_b

    # Read current states of both channels
    current_state_a = GPIO.input(ENCODER_A_PIN)
    current_state_b = GPIO.input(ENCODER_B_PIN)

    # Determine direction based on state transitions
    if last_state_a == 0 and current_state_a == 1:  # Rising edge on A
        if current_state_b == 0:  # B is low, moving clockwise
            position += 1
        else:  # B is high, moving counterclockwise
            position -= 1
    elif last_state_a == 1 and current_state_a == 0:  # Falling edge on A
        if current_state_b == 1:  # B is high, moving clockwise
            position += 1
        else:  # B is low, moving counterclockwise
            position -= 1
```

### Absolute Encoder

An absolute encoder has a coded encoder disk. It comprises of a few concentric tracks (circles), each track has a hollow (0) and a slot (1). This way, we can read the absolute position of an encoder disk.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/beeaece3-4dc2-41a4-b238-aa6ebe181dfc" height="300" alt=""/>
        <figcaption><a href="https://www.akm.com/global/en/products/rotation-angle-sensor/tutorial/type-mechanism-2/">Source</a></figcaption>
    </figure>
</p>
</div>

### Capacitors

- In betwen plates of a capacitor, we need dielectric. It's an insulating material that increases the capacitor's ability to store charge. Dielectrics include: air, vacuum, ceramic, electrolytic, etc.
  - In an ideal dielectric there is no current flow.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/G3JHT1V5/capacitordiagram.png" height="300" alt=""/>
        <figcaption><a href="https://www.physics-and-radio-electronics.com/electronic-devices-and-circuits/passive-components/capacitors/capacitorconsutructionandworking.html">Source: physics-and-radio-electronics.com </a></figcaption>
    </figure>
</p>
</div>

- A capacitor gathers charge via displacement current, not conduction current across the plates:
    1. You connect a voltage source (like a battery) across the plates.
    2. Electrons accumulate on one plate (negative terminal of the source).
    3. This creates an electric field through the dielectric.
    4. That electric field repels electrons from the opposite plate (pulling positive charge toward it).
    5. No electrons physically cross the dielectric — they just pile up on each side.

### EC (Electrically Commuted Motors) vs DC Motors

TODO

## Electronics Testing

### ICT (In-Circuit Testing)

ICT is a type of testing performed directly on PCB to check each component for open / short circuits, incorrect part placement. Method:

- Build a "bed-of-nails" fixture that includes test points, probes
- The fixture measures resitance, voltage, capacitance

So it's quick, good test coverage, automated. but it can't test the high level features.

### Functional Testing

Test the system's overall performance, such as communication, user interfaces, power distribution.

## MEMS IMU

### Accelerometer

When linear acceleration occurs, an inertial mass suspended inside the MEMS structure displaces relative to fixed electrodes. This displacement changes the capacitance between the mass and surrounding plates. By measuring these changes, we can infer the applied linear acceleration.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/MHjLBdQt/q5-9il.gif" height="300" alt=""/>
        <figcaption><a href="https://makeagif.com/gif/how-mems-accelerometer-gyroscope-magnetometer-work-arduino-tutorial-q5-9il">Source</a></figcaption>
    </figure>
</p>
</div>

### Gyro

Gyroscopes use the Coriolis effect. The Coriolis force arises in a rotating reference frame and causes moving objects to appear to deviate from a straight-line path. For example, a ball rolling straight in an inertial frame appears to curve when observed from a rotating disk.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/Gh4PFtRp/Coriolis.gif" height="300" alt=""/>
        <figcaption><a href="">Source: </a></figcaption>
    </figure>
</p>
</div>

With stationary frame travelling direction of the object `v` and angular velocity `w`, the perceived Coriolis Force is

$$
\begin{gather*}
\begin{aligned}
& F = 2m \vec{v} \times \vec{w}
\end{aligned}
\end{gather*}
$$

And the Coriolis force is perpendicular to both `v` and `w`

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/YvWy662K/2025-05-09-12-45-47.png" height="200" alt=""/>
        <figcaption><a href="https://www.youtube.com/watch?v=PK05u9c3yWI">Source</a></figcaption>
    </figure>
</p>
</div>

A MEMS gyroscope often uses a 'tuning fork' design with two proof masses vibrating in opposite directions. Each mass is suspended by springs and forms capacitors with fixed side plates. When the device rotates, Coriolis forces cause the masses to deflect perpendicular to both their vibration and the rotation axis. This results in a measurable change in capacitance, with a net difference:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/GHN5T7Cx/2025-05-09-12-46-04.png" height="200" alt=""/>
        <figcaption><a href="https://www.youtube.com/watch?v=PK05u9c3yWI">Source</a></figcaption>
    </figure>
</p>
</div>

which is proportional to angular velocity.

If linear acceleration occurs along the sensing axis, both masses are displaced in the same direction, resulting in opposing changes in capacitance. The net differential change is:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/R0J0SWjj/2025-05-09-12-56-01.png" height="300" alt=""/>
        <figcaption><a href="https://www.youtube.com/watch?v=PK05u9c3yWI">Source</a></figcaption>
    </figure>
</p>
</div>

allowing the system to distinguish linear acceleration from angular rotation.

## Telecommunication

### Acoustic Modem

An acoustic model converts `digital -> acoustic signal -> demodulate signal`
