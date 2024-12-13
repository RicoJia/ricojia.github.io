---
layout: post
title: Deep Learning - Speech Recognition Hands On
date: '2022-03-29 13:19'
subtitle: GRU-Based Trigger Word Detection
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
    - Hands-on
---

## Trigger Word Detection

**Goal**: we can the word "activate" and hear a chime.
**Data**: the data was recorded at various venues such as libraries, cafes, restaurants, homes, and offices. It has a positive set, which includes the trigger word "activate", a background noise set, and a negative set that includes other words.

- An audio clip is a serious of numerical recordings of air pressure. In this case, if we record with a 44100Hz mic, we will get 44100 numbers.
- It's usually quite hard to detect keywords directly, so we use a **spectogram** which tells distribution of frequencies at any given time. A spectogram is calculated using Fast-Fouerier Transform FFT. Without going deep into FFT, the way to generate a spectogram is:
  - Split the audio into overlapping windows along the time axis. The shorter the window, the better the time-domain accuracies. The longer the window, the better the freuqency-domain accuracies.
  - Compute magnitude of frequencies using FFT on each time window. Since the time windows overlap, the frequency domain components across domains are more consistent.
  - For example, in the below figure, each column represents a time window. Green means active, blue means not-active.

    ![2024-11-14 12-19-41屏幕截图](https://github.com/user-attachments/assets/4239238d-ad20-4aa6-85af-0b063b55bf08)

- In our case, we choose to work on 10s clips. The mic is 44100 Hz, so the raw input is `Tx=441000`. On the spectogram, there are 5511 time windows, with 101 frequency windows, so the spectogram input is `(101, 5511)`

### `pydub` For Audio Synthesis

- Needs `sudo apt install ffmpeg`
- To generate a simple clip,

```python
#!/usr/bin/env python3
from pydub import AudioSegment
from pydub.generators import Sine
from pydub.playback import play
# Generate a 440 Hz sine wave tone for 2 seconds
tone_440 = Sine(440).to_audio_segment(duration=2000)
tone_550 = Sine(550).to_audio_segment(duration=1500)
tone = tone_440.overlay(tone_550)
silence = AudioSegment.silent(duration=1000)
tone_550_weak = tone_550 - 6
concat_tone = tone_440 + silence + tone_550 + tone_550_weak
# play(tone)
play(concat_tone)
```

- `pydub` uses 1ms as time step, so a 10s clip is always 10000 time steps.

### Data Synthesis

- It's quite slow to record 10s audio clips and recognize when the positive and negative words appear. So we can record positive and negative words, download some background clips, then add them to the words.
- Label $y^{(t)}$ is when the word "activate" is done. Also, **to more have a more balanced dataset (with background)**, we add the `label=0` for 50 consecutive timesteps, 1 time step **after** the word "activate" is done.
- We want to create a dev set that's similar to the actual test set. So we want to make sure the two's distributions are similar. **In this case, I'm using real audio instead of synthesized audio**

### Model

An end-to-end deep learning approach could be used to build an effective trigger word detection system.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/251c666d-411c-4717-9086-6cc88283b509" height="600" alt=""/>
       </figure>
    </p>
</div>

- As stated above, the input is the spectogram, a `(Tx = 5511, F=101)` 2D vector.
- The output is a binary classification result of 1375  time steps, a `(Ty=1375)` 1D vector.
- The 1D conv layer can extract low level features, so the GRU only needs to work on 1375 time steps. **This is a common pre-processing step for audio data before passing into an RNN/GRU/LSTM**.
- The model uses a uni-directional GRU architecture because we think the **previous information** is more important than the future information. This greatly reduces the model complexity.
- **In a real application**, one can perform this detection using a sliding 10s window **every second**, so we get a more real-time response.

- How to apply 1D conv on 2D data?
  - We actually extract the 1D frequency vector of each time window and pass it into the 1D conv layer.
  - The 1D Conv layer is actually a `TimeDistributed` Layer. Its expects an input size `(batch_size, time_steps, freq_bins)`.
    - So we don't need to manually decompose the input into time vectors and pass them in individually
- Why drop out with such a high rate?
  - In a small dataset, it's quite easy to overfit
  - A high dropout rate forces the network to rely less on specific neurons, but more on the distributed patterns across many neurons. This could promote generalization
- TODO: Why 2 dropouts, one before BN, one after? This is very aggressive and I've never seen it used else where.
  - Normally, `Dropout` after BN is more preferred.
- The final dense layer is also "smart" for our application. It expects an input size `(batch_size, d0, d1)`, then it creates a kernel with shape `(d1, units)`. In our case `d0` is time, and we want to extract features along d1. Great!

How does a [Time Distributed Layer works](https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/)?

- A time distributed layer is a **wrapper** that is applied on a specific layer, that is, **Dense, etc.**. It applies the same set of parameters of the layer to all time steps. It outputs a prediction per timestep. **It's actually a convenient way to frame one-to-one sequential models**
- In our case, we will be using `TimeDistributed(Dense(...))`
- In practice, vectors for each timestep is processed in parallel.
- The input is at least 3D: `(batch size, time steps, feature dim)`

### Model Summary

```bash
Model: "functional_6"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_5 (InputLayer)         [(None, 5511, 101)]       0         
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 1375, 196)         297136    
_________________________________________________________________
batch_normalization_12 (Batc (None, 1375, 196)         784       
_________________________________________________________________
activation_4 (Activation)    (None, 1375, 196)         0         
_________________________________________________________________
dropout_16 (Dropout)         (None, 1375, 196)         0         
_________________________________________________________________
gru_8 (GRU)                  (None, 1375, 128)         125184    
_________________________________________________________________
dropout_17 (Dropout)         (None, 1375, 128)         0         
_________________________________________________________________
batch_normalization_13 (Batc (None, 1375, 128)         512       
_________________________________________________________________
gru_9 (GRU)                  (None, 1375, 128)         99072     
_________________________________________________________________
dropout_18 (Dropout)         (None, 1375, 128)         0         
_________________________________________________________________
batch_normalization_14 (Batc (None, 1375, 128)         512       
_________________________________________________________________
dropout_19 (Dropout)         (None, 1375, 128)         0         
_________________________________________________________________
time_distributed_2 (TimeDist (None, 1375, 1)           129       
=================================================================
Total params: 523,329
Trainable params: 522,425
Non-trainable params: 904
```

## Training And Prediction

Trigger word detection takes a long time to train.

```python
opt = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
```

For tasks with a lot of backgrounds, accuracy is not the best metric. F1 Score, precision & recall could be better scores.
