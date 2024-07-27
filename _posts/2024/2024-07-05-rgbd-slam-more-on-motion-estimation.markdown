---
layout: post
title: More On Motion Estimation 
date: '2024-07-04 10:11'
excerpt: 5 Point Algorithm, How to solve DLT, and 8 point algorithm
comments: true
---

## Why Do We Still Need Multi-View Geometry, In 2024?

LLMs are everywhere now. However, they are rarely seen in 3D scene reconstruction except for those on depth learning from single images. Dong et al. claim that is because mismatching feature points are inevitable, and once they are passed into a deep neural net, there is no such an equivalent to RANSAC. This results in large noises in reconstruction errors.

Advantages of Multi-view geometry mainly include: clarity in mechanism and high accuracy. Disadvantages include: computation complexity, and high dependence on texture.

## Degeneration Of The Essenstial Matrix, E

$E=t \times R$. E's rank can drop when 1. the baseline distance between cameras approaches 0, or 2. matching feature points are co-planar. Here is why:

1. When all feature matches are co-planar, though E itself does not depend on the features, the way we estimate does. Recall:

$$
\begin{gather*}
\begin{pmatrix}
u_2^1 u_1^1 & u_2^1 v_1^1 & u_2^1 & u_2^2 u_1^1 & u_2^2 v_1^1 & u_2^2 & v_2^1 u_1^1 & v_2^1 v_1^1 & v_2^1 & v_2^2 u_1^1 & v_2^2 v_1^1 & v_2^2 & u_1^1 & v_1^1 & 1 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
u_2^8 u_1^8 & u_2^8 v_1^8 & u_2^8 & u_2^8 u_1^8 & u_2^8 v_1^8 & u_2^8 & v_2^8 u_1^8 & v_2^8 v_1^8 & v_2^8 & v_2^8 u_1^8 & v_2^8 v_1^8 & v_2^8 & u_1^8 & v_1^8 & 1 \\
\end{pmatrix} e = Ue = 0
\end{gather*}
$$

$U$ is $8 \times 9$ and normally should be of rank 8. This makes sure E is up to a scale and has 1 degree of freedom (so $E$ has 2dof). But when a homography exists all the feature points, 
When homography exists between all feature matches, more constraints could be added to $U$. E.g., 

$$
\begin{gather*}
p_2 = H p_1 
\\
=>
\\
p_2^TEp_1 = p_1 H^T E p_1 = 0 
\end{gather*}
$$

The proof was more involved than I thought (for those who are interested, see [2]). Basically, one could prove it from the perspective of $U$'s nullity. When the nullity is 3, that is there are 3 linearly independent solution to E, the above degeneration could happen.

2. When the baseline distance between cameras approaches 0, intuitively, $E = txR$ would approach 0. But when we estimate, the $U$ matrix (see above) will also have a nullity of 3.


So, one way to combat this is using DLT to calculate homography aside from applying 8-point algorithm, and choose the result with minimum reprojection error.

## References

[1] Dong, Q., Shu, M., Cui, H., Xu, H., & Hu, Z. (2018). Learning stratified 3D reconstruction. *Science China Information Sciences*, 61(2), 023101. DOI: https://doi.org/10.1007/s11432-017-9234-7

[2] P. H. S. Torr, A. Zisserman, and S. J. Maybank. "Robust Detection of Degenerate Configurations whilst Estimating the Fundamental Matrix." Robotics Research Group, Department of Engineering Science, Oxford University, Department of Computer Science, Reading University, UK. DOI: https://www.robots.ox.ac.uk/~vgg/publications/1998/Torr98c/torr98c.pdf