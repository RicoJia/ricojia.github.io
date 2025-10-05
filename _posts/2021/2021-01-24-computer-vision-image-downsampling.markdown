---
layout: post
title: Computer Vision - Ray Tracing, Ray Casting, Gaussian Splatting
date: '2021-01-27 13:19'
subtitle: 
comments: true
tags:
    - Computer Vision
---

## Introduction

In the 16th century one innovative artist, Albrecht Düre created a “Perspective Machine” to help artists draw perspectives accurately. He did this by creating a screened 2D frame, between the artist and the drawing subject. The artist would then establish a line of sight from the artist’s eye through the 2D screen to any part of the drawing subject. In front of him was his drawing paper with a matching grid. (From [Bob Duffy](https://medium.com/sideofcyber/side-of-ray-tracing-c4721bf9bba8))

![img](https://i.postimg.cc/gcDCsp4C/Screenshot-from-2025-10-05-07-41-49.png)

This technique for rendering an image by tracing the path of light through cells of a 2D image plane is called ray casting or ray tracing, and it’s how today’s advanced computer graphics got its start. (From [Bob Duffy](https://medium.com/sideofcyber/side-of-ray-tracing-c4721bf9bba8))

![img](https://i.postimg.cc/ZnVX5P0s/Screenshot-from-2025-10-05-07-32-21.png)

This is mimicking the physics of optics but in reverse. In our world, light rays start from the environment, bounce of objects, then land in our eye. For rendering, we project rays in reverse, because we only need the rays that land in the camera. (From [Bob Duffy](https://medium.com/sideofcyber/side-of-ray-tracing-c4721bf9bba8)). This allows us simulate how light travels in the physical world. Due to reflection and refraction, we can even paths of simulate secondary and tertirary beams

The issue about ray tracing is it's too slow. Most 3D applications run at about 24–60FPS. A technique called rasterization is invented: an object is decomposed into triangles that intersect together.

![img](https://i.postimg.cc/sf0024ck/rasterization-vs-raytracing-l.jpg)

The rasterization pipeline is in short as follows ([A good video that explains the rasterization pipeline](https://youtu.be/brDJVEPOeY8)):

1. Inputs: vertices (3D points) + triangle connectivity (triangle list). Optional: surface normals, UVs (2D position on the texture image for representing texture)
2. Vertex shading:
    1. Coordinate transformation: Model → World → View (camera extrinsics) → Projection (camera intrinsics/perspective), this is called "clip space" (x, y, z, w). Clipping against view frustum happens here.
3. Tesselation (optional): . It subdivides patches based on tessellation control/evaluation shaders;
4. (Optional) Geometry shader: can add/remove/modify primitives.
5. Primitive assembly: uses your triangle list/strip indices to form triangles.
6. After projection you do perspective divide (to Normalized Device Coordinates, NDC) → viewport transform (to screen pixels).
7. Rasterization: coverage test converts each triangle to fragments (candidate pixels); Rasterization turns triangles into pixels (fragments) and uses depth to resolve visibility

One issue with rasterization is it does not reproduce the shading effect very well. (Source: [Simon Kang](https://medium.com/@simon.kong95/what-you-need-to-know-about-ray-tracing-and-rasterization-b7a4b1489215)) For real-time ray tracing in video games, you’ll need advanced hardware. For gamers wanting that extra dose of reality, it may be a worthy investment.

![img](https://i.postimg.cc/1X1J1vPf/Screenshot-from-2025-10-05-07-59-47.png)

Unlike Ray Tracing which simulates reflections and refractions, (volumetric) ray casting simply shoots one beam through a 3D volume. The goal is to accumulate color and opacity at each pixel. Surface Ray casting just shoots one beam and determines and returns if there's hit /no hit from the surface volume (collision detection)
