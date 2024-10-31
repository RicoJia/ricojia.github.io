---
layout: post
title: Deep Learning - Deeplab Hands On
date: '2022-08-01 13:19'
subtitle: Deeplab
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Motivation

UNet has restricted receptive field. It is sufficient for identifying local areas such as tumors (in medical images). When we learn larger image patches, UNet was not sufficient.

DeepLab V1 - V3 enlarges its receptive fields by introducing "dilated convolution" (空洞卷积)
