---
layout: post
title: More On Motion Estimation 
date: '2024-07-05 10:11'
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

## Five Point Algorithm

**Motivation:** Since E has 5 dof, why don't we do use five points? The reason for using fewer feature matches points is from feature matches, there could be many bad matches. People are using RANSAC to get more robustness, adn the five point algorithm lowers the chance of using bad matches. (outliers).

The basic set up is [the same as the 8-point algorithm](https://ricojia.github.io/2024/07/04/rgbd-slam-motion-estimation-from-epipolar-constraints.html)

$$
\begin{gather*}
[u_2 u_1 , u_2 v_1 , u_2 , v_2 u_1 , v_2 v_1 , v_2 , u_1 , v_1 , 1] · e = Ae = 0
\end{gather*}
$$

With 5 feature matches, we get 5 equations. So $U$ should have 4 linearly independent solutions: `X, Y, Z, W`. The final solution $E$ must be a linear combination of them. Since $E$ is up to scale and has 1 dof, we make $c_{w}=1$

$$
\begin{gather*}
E = c_x X + c_y Y + c_z Z + c_w W \tag{1}
\end{gather*}
$$

Then, Nister et al. found two properties of E that can be used as constraints: [3]

$$
\begin{gather*}
\text{det}(\mathbf{E}) = 0 \tag{2}
\\
\mathbf{E}\mathbf{E}^T\mathbf{E} - \frac{1}{2} \text{trace}(\mathbf{E}\mathbf{E}^T)\mathbf{E} = \mathbf{0} \tag{3}
\end{gather*}
$$

Plugging (1) into (2) and (3) gives 3 cubic equations in 3 variables , which have at most 9 valid equations. However, **People have widely adopted the five point algorithm because these equations can be quickly eliminated, using redundant feature matches**

## References

[1] Dong, Q., Shu, M., Cui, H., Xu, H., & Hu, Z. (2018). Learning stratified 3D reconstruction. *Science China Information Sciences*, 61(2), 023101. DOI: https://doi.org/10.1007/s11432-017-9234-7

[2] P. H. S. Torr, A. Zisserman, and S. J. Maybank. "Robust Detection of Degenerate Configurations whilst Estimating the Fundamental Matrix." Robotics Research Group, Department of Engineering Science, Oxford University, Department of Computer Science, Reading University, UK. DOI: https://www.robots.ox.ac.uk/~vgg/publications/1998/Torr98c/torr98c.pdf

[3] Nistér, D. 2004. An Efficient Solution to the Five-Point Relative Pose Problem. In *Proceedings of the 2004 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'04)*, Vol. 2. IEEE, 195–202. DOI: https://www-users.cse.umn.edu/~hspark/CSci5980/nister.pdf