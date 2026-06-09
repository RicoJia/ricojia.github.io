---
layout: post
title: Computer Vision - Tessellation
date: 2021-01-30 13:19
subtitle:
comments: true
tags:
  - Computer Vision
---
## Tessellation

**Tessellation** means converting smooth CAD surfaces into a triangle mesh. **Linear deflection** controls how far the triangle mesh is allowed to deviate from the true CAD surface. Smaller linear deflection = more accurate mesh, more triangles, larger OBJ. Example: `linear deflection = 0.10 mm`.  means the generated triangle surface should stay within roughly **0.10 mm** of the original CAD surface. **Angular deflection** controls how much the triangle normals are allowed to change between adjacent pieces of a curved surface. Smaller angular deflection = smoother-looking curved/rounded surfaces, more triangles. example: `angular deflection = 0.25 rad ≈ 14.3°`

![](https://raw.githubusercontent.com/cwant/tessagon/master/documentation/images/tessagon_demo.gif)
Tesslleating 3D meshes into triangles, hexagons, and other shapes
