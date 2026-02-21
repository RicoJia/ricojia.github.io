---
layout: post
title: "[ML] -Point-cloud-compression-2-FPFH-Features"
date: 2025-01-08 13:19
subtitle:
header-img: img/post-bg-o.jpg
tags:
  - Machine-Learning
comments: true
---


## Other Alternatives

- Chamfer distance  
- Earth Mover's Distance (EMD)  
- Normal consistency loss  
- Curvature loss  
- Learned feature-space loss (via PointNet / DGCNN embeddings)  

## What is FPFH
  
**FPFH (Fast Point Feature Histograms)** describe local geometric structure using histograms of normal relationships.  
  
### Potential advantages  

- Captures **local geometry structure**  
- More invariant to small spatial shifts  
- Encodes curvature and surface shape  
- Could penalize structural distortion rather than just point displacement  
  
### Potential problems  
  
1. **Non-differentiability**  

- FPFH involves:  
- Normal estimation  
- Neighborhood selection  
- Histogram binning  
- Histogram binning is not naturally differentiable.  
- Nearest-neighbor search for features adds more non-smooth operations.  
  
2. **Instability**  

- Normals are sensitive to noise.  
- Small geometry changes may cause large feature changes.  
  
3. **Heavy computation**  

- Much more expensive than Chamfer distance.  
