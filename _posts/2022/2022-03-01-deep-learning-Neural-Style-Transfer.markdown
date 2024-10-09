---
layout: post
title: Deep Learning - Neural Style Transfer
date: '2022-03-01 13:19'
subtitle: What Conv Net Is Learning
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## What Do Conv Nets Learn? 

For the ease of explanation, below I will use an example, where a conv layer has 3 input channels, and 5 output channels. As a recap: this layer has 3 filters, each filter is a **3D kernel kernel**: `HxWx5`; the output feature map of a filter is the sum of result of the input passing through the 3 kernels. An activation or neuron in the context of CNN refers to an element in an output feature map.Each activation can "see" a region in the initial input images, the region is called "receptive field". 

Because all color channels are considered, a kernel itself learns one type of **feature**. It could be a pattern like an edge, a corner, or a texture like a certain curve in a certain color. So, **high activations on a feature map correponds to the existence of the learned features of the corresponding kernels, in those spatial locations.**

Zeiler et al. proposed a "Deconvnet" that identifies the receptive field in an input that leads to most activation on the output feature map of a conv layer [1]. TODO: How?

In the experiment result, we can see that the shallow layers learn local features such as edges, corners, due to small receptive fields. Mid layers learn contours, shapes (like squares), and color combinations. Deep layers have large receptive fields and can learn complex shapes like faces, semantic concepts

![Screenshot from 2024-10-08 17-47-18](https://github.com/user-attachments/assets/d136c6f0-1069-4f05-b515-de87906cbf46)


## Neural Style Transfer 

### Similarity of Styles

One great intuition from Gatys et al's work is how to extract "style" from an image. **A style can be roughly summarized as having a set of patterns in the same spatial windows across an image.** Some cases include: caligraphers have their own distinct strokes, painters perfer certain patterns in specific colors. With that in mind, the "style" of an image can be measured by:

1. Passing the image through a series of conv layers. 
2. Take the output feature maps from a given layer. 
3. Multiply **all pairs of feature maps `F` together**, then add them together. This will get us a style matrix, or **gram matrix**, `G(I)`. For an element `G(I)_{i,j}`, across all channels `K`,

$$
G(I)_{i,j} = \sum_K F_i \cdot F_j
$$

Note:
- The diagonal elements in the Gram matrix measures how "active" feature maps are to the learned features in their filters. For example, if a filter learns vertical lines, the feature map will show if there are many vertical lines at each spatial locations
- The non-diagonal ones measure the similarity. 

Then, **dissimilarity** of two images' styles is equivalent to detecting the similarity of their activations to a given set of learned features spatially. That can be done by comparing their gram matrices

$$
J_{style}(I1, I2) = \frac{1}{(2n_H n_W n_C)^2}|G(I1) - G(I2)|
$$

The distance metric being in use is the "Squared Frobenius Norm". Also note that there's a normalizing factor in the style cost.

Actually, if we put the results from two layers together, we get a more comprehensive result:

$$
J_{style}(S,G) = \sum_L \lambda^l J^l_{style}(I1, I2)
$$

## How To Find the Similarity of Content

Between two images, it's relatively intuitive to find the similarity of their contents at layer `l`.

$$
J_{content}(I1, I2) = \frac{1}{2} \sum_I (F^l(I1) - F^l(I2))^2
$$

### Putting Content And Style Together

Gatys et al proposed a method to generate a picture that resembles content C, but in the style of S.

![Screenshot from 2024-10-08 16-58-18](https://github.com/user-attachments/assets/b6eafb34-cca8-4433-98d3-0fc0d166b4e2)

- This is done by crafting a cost function, with `G` being the generate dimage, `C` being the original image, and `S` being the styleimage. Some may argue that $\alpha$ and $\beta$ are redundant, but those are the notations used by Gatys et al.

$$
\begin{gather*}
J(G) = \alpha J_{content}(C, G) + \beta J_{style}(S, G)
\end{gather*}
$$

- The 2nd step is to generate a random image.
- The 3rd step is to do gradient descent on the network with the current image using the cost function. 
- Over iterations, the images should look like below: 

![Screenshot from 2024-10-08 17-57-49](https://github.com/user-attachments/assets/1790ce15-3327-4ebb-890e-f97fb11a571c)

- In neural style transfer, we train the pixels of an image, and not the parameters of a network. How? 
    1. Generate a random image $I$
    2. Have a conv net like VGG-19 as the backbone for feature extraction
    3. Calculate Gram Matrix,  the loss $J(C,S)$, and the partial derivatives $\frac{\partial L}{\partial I}$ through back propagation
    4. Update image value by $I = I - \lambda \frac{\partial L}{\partial I}$


## Reference

[1] [Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. In European Conference on Computer Vision (ECCV) (pp. 818-833). Springer. https://doi.org/10.1007/978-3-319-10590-1_53](https://arxiv.org/abs/1311.2901) 

[2] [Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). A neural algorithm of artistic style. Journal of Vision, 16(12), 326. https://doi.org/10.1167/16.12.326](https://arxiv.org/pdf/1508.06576)
