---
layout: post
title: Deep Learning - Speech Recognition
date: '2022-04-03 13:19'
subtitle: Audio Signal Processing, Spectogram
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Overview

In speech recognition, initially scientists thought that phonemes, like the invididual sounds in words (like "g" "v" in "give") were the best way to represent audio words. This was because, human ears can recognize the intensity of sounds at different frequencies. This is similar to applying a **spectrogram**, where the x axis is time, y axis is frequencies, and colors represent intensities. Most speech recognition systems are Hidden Markov Models. Phonemes reduce the state space greatly, so scientists hypothesized that human ears would recognize speeches based on them as well.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/f5a15e25-1c2e-4026-b39c-41ba22fd4cf9" height="300" alt=""/>
       </figure>
    </p>
</div>

One architecture is to use an RNN and output phonemes one at a time
<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/10eb56e8-6b7b-4561-98e7-ac2c35ee3d4a" height="300" alt=""/>
       </figure>
    </p>
</div>

In speech recognition texts, there are usually a lot more "background" than actual speeches. So if your time step is small, you might hear "the-e-e-e-e k-k-k-w-w-u-i-k-k-k....". The above model does not merge the consecutive identical phonemes. In the Connectionist Temporal Classification (CTC) model, the ML model learns a "blank letter" that represents background. Also, CTC will combine the same consecutive letters together so it will have the output "thekwuik" in the above case. Another example is `__c_oo_o_kk___b_ooooo__oo__kkk -> cookbook`

### Trigger Word Detection

On top of the RNN schemes above, training a trigger word detection neural net requires a dataset with labels 0 = background, 1=trigger word. One issue is imbalanced labels because there are **always more labels than trigger words**. Each `x^{(t)}` represents the features of the audio at time.
