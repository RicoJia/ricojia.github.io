---
layout: post
title: "[ML] -Point-cloud-compression-1-Basic-Concepts"
date: 2026-01-05 13:19
subtitle: PSNR, Chamfer Distance
header-img: img/post-bg-o.jpg
tags:
  - Machine-Learning
comments: true
---
## [Part 1] PSNR vs SNR

In signal processing, the **Signal-to-Noise Ratio (SNR)** is defined as:  
  
$$  
\mathrm{SNR} = 10 \log_{10} \left( \frac{\mathrm{Var}(\text{signal})}{\mathrm{Var}(\text{noise})} \right)  
$$  
  
- Unit: **dB**  
- Measures the ratio between signal power (variance of the signal) and noise power (variance of the error).

The **Peak Signal-to-Noise Ratio (PSNR)** replaces signal variance with the squared peak signal value:  
  
$$  
\mathrm{PSNR} = 10 \log_{10} \left( \frac{\mathrm{MAX}^2}{\mathrm{MSE}} \right)  
$$  
  
- Unit: **dB**  
- $\mathrm{MAX}$ is the maximum representable signal value.  
- $\mathrm{MSE}$ is the mean squared error between reference and reconstructed data.

For point cloud geometry:  
  
- $\mathrm{MAX}$ = diagonal length of the 3D bounding box of the reference point cloud.  
- $\mathrm{MSE}$ is computed using nearest-neighbor distances.  
  
**Procedure:**  
  
1. For each point $p_i$ in the reference point cloud, find its nearest neighbor $\hat{p}_i$ in the reconstructed cloud.  
2. Compute:  
  
$$  
\mathrm{MSE} =  
\frac{1}{N}  
\sum_{i=1}^{N}  
\left\| p_i - \hat{p}_i \right\|^2  
$$  
Colored PSNR: For attribute (color) distortion:  
  
- $\mathrm{MAX} = 255$ (for 8-bit color channels).  
- MSE is computed across RGB channels and averaged:  
  
$$  
\mathrm{MSE} =  
\frac{1}{3N}  
\sum_{i=1}^{N}  
\left[  
(R_i - \hat{R}_i)^2 +  
(G_i - \hat{G}_i)^2 +  
(B_i - \hat{B}_i)^2  
\right]  
$$  
  
where:  

- $(R_i, G_i, B_i)$ are original colors,  
- $(\hat{R}_i, \hat{G}_i, \hat{B}_i)$ are reconstructed colors.  
  
PSNR is then computed as:  
  
$$  
\mathrm{PSNR} =  
10 \log_{10}  
\left(  
\frac{255^2}{\mathrm{MSE}}  
\right)  
$$

---

## [Part 2] Loss - Chamfer Distance
  
A common geometry loss for point clouds is the **Chamfer Distance (CD)**.  
  
Given two point sets:  

- Reference point cloud: $P = \{p_i\}_{i=1}^N$  
- Reconstructed point cloud: $\hat{P} = \{\hat{p}_j\}_{j=1}^M$  
  
The (symmetric) point-to-point Chamfer distance is:  
  
$$  
\mathcal{L}_{\text{CD}}(P, \hat{P}) =  
\frac{1}{N} \sum_{i=1}^{N}  
\min_{\hat{p} \in \hat{P}} \| p_i - \hat{p} \|^2  
+  
\frac{1}{M} \sum_{j=1}^{M}  
\min_{p \in P} \| \hat{p}_j - p \|^2  
$$  
  
This is essentially the average squared nearest-neighbor distance in **both directions**.

Is Chamfer Distance Differentiable?  **Short answer**:  
**Yes, almost everywhere — and it is widely used in gradient-based optimization.**  
  
- The squared Euclidean distance $\|p - \hat{p}\|^2$ is fully differentiable.  
- The only non-smooth operation is the **nearest neighbor selection** (the $\min$).  

### Step 1: Forward Pass Nearest-neighbor (NN) assignment  

Define the (squared-distance) NN index for each reference point:  
$$  
j^*(i) \;=\; \arg\min_{j} \|p_i - \hat p_j\|^2.  
$$  
  
Then the loss can be written *with the discrete assignment made explicit*:  
$$  
\mathcal{L}_{P\to \hat P}  
=  
\frac{1}{N}\sum_{i=1}^{N} \|p_i - \hat p_{j^*(i)}\|^2.  
$$  
This is the key idea: **during the backward pass we treat the current NN assignments as fixed** (they are recomputed each forward pass, but within a given forward/backward evaluation they are constants).  You can compute the Jacobian with respect to point positions:  
  
$$  
\frac{\partial}{\partial \hat{p}} \| p - \hat{p} \|^2  
=  
2(\hat{p} - p)  
$$  
  
So in the graph?, the jacobian of the final cost w.r.t a contributing point is:  
$$  
\frac{\partial \mathcal{L}_{P\to \hat P}}{\partial \hat p_k}  
=  
\frac{1}{N}\sum_{i\in S_k} 2(\hat p_k - p_i).  
$$

### Step 2: Chamfer Distance — Gradient w.r.t. Reconstructed Points  
  
The reconstructed point cloud $\hat{P} = \{\hat{p}_j\}_{j=1}^M$ is the output of the network, so during backpropagation we compute gradients with respect to these points.  
  
The symmetric Chamfer Distance between the original point cloud $P = \{p_i\}_{i=1}^N$ and the reconstructed point cloud $\hat{P}$ is:

$$  
\mathcal{L}_{\text{CD}}(P,\hat P)  
=  
\frac{1}{N}\sum_{i=1}^{N} \min_{j} \|p_i - \hat{p}_j\|^2  
+  
\frac{1}{M}\sum_{j=1}^{M} \min_{i} \|\hat{p}_j - p_i\|^2.  
$$
To compute the gradient with respect to a reconstructed point $\hat{p}_k$, we consider the second term (reconstructed → original). Let $S_k$ be the set of original points for which $\hat{p}_k$ is the nearest neighbor. Then the gradient is:  
  
$$  
\frac{\partial \mathcal{L}_{\hat{P} \to P}}{\partial \hat{p}_k}  
=  
\frac{1}{M}  
\sum_{i \in S_k}  
2(\hat{p}_k - p_i).  
$$  

This gradient pushes each reconstructed point toward the original points for which it is the closest match, ensuring the predicted cloud aligns with the ground truth during training.

### Step 3: Backward Pass — Jacobian View (Aggregation)  
  
Stack reconstructed points into a single vector:  
  
$$  
\hat{\mathbf{x}} =  
[\hat p_1^\top,\hat p_2^\top,\dots,\hat p_M^\top]^\top  
\in \mathbb{R}^{3M}.  
$$
For a single term $\|p_i - \hat p_{j^*(i)}\|^2$, the gradient w.r.t. $\hat{\mathbf{x}}$ is zero everywhere except in the block corresponding to $j^*(i)$:  
  
- For block $k = j^*(i)$:  

$$  
\frac{\partial}{\partial \hat p_k}  
\|p_i - \hat p_{j^*(i)}\|^2  
=  
2(\hat p_k - p_i)  
$$

- For blocks $k \neq j^*(i)$:  

$$  
\frac{\partial}{\partial \hat p_k}  
\|p_i - \hat p_{j^*(i)}\|^2  
=  
0  
$$
Thus, the Jacobian is **block-sparse**: each reference point contributes to exactly one reconstructed point (for this one-way term).

### Step 4: Aggregating Over All Points  
  
Now summing over all reference points in the $P \to \hat P$ term,  
  
$$  
\mathcal{L}_{P \to \hat P}  
=  
\frac{1}{N}  
\sum_{i=1}^{N}  
\|p_i - \hat p_{j^*(i)}\|^2,  
$$
the gradient for a reconstructed point $\hat p_k$ becomes:  
  
$$  
\frac{\partial \mathcal{L}_{P \to \hat P}}{\partial \hat p_k}  
=  
\frac{1}{N}  
\sum_{i:\, j^*(i)=k}  
2(\hat p_k - p_i).  
$$  
  
Define the index set  
  
$$  
S_k = \{\, i \mid j^*(i) = k \,\}.  
$$  
  
Then we can write compactly:  
  
$$  
\frac{\partial \mathcal{L}_{P \to \hat P}}{\partial \hat p_k}  
=  
\frac{2}{N}  
\sum_{i \in S_k}  
(\hat p_k - p_i).  
$$

So aggregation simply amounts to **summing all contributions from reference points that selected $\hat p_k$ as their nearest neighbor**.  
  
### Q & A: Where non-differentiability comes from (and why it’s “almost everywhere”)  
  
The only non-smooth part is:  
  
$$  
j^*(i) = \arg\min_j \|p_i-\hat p_j\|^2.  
$$  
  
As you move $\hat p_j$'s continuously, most of the time the identity of the minimizer stays the same, so $j^*(i)$ is constant locally and the loss is a smooth quadratic in that region.  
  
A change happens only when two candidates tie:  
$$  
\|p_i-\hat p_a\|^2 = \|p_i-\hat p_b\|^2,  
$$  
which defines a boundary of measure zero in continuous space. That’s the “assignment switch” event.  
  
At those boundaries:  

- the gradient is not uniquely defined (there are multiple minimizers),  
- but optimization still works because you almost never land exactly on the boundary, and even if you do, any small perturbation picks a side.  
