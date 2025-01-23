---
layout: post
title: Electronics - General Electronics Notes
subtitle: Encoders, Motors, Testing
date: '2017-06-04 13:19'
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