---
comments: true
tags:
header-img: img/post-bg-os-metro.jpg
subtitle: Checker Board Detection
date: 2024-01-06 13:19
title: Computer Vision - Uzzle Solver
layout: post
---
## FindContour

A **contour** is a sequence of points that lie along the **boundary** of a connected region (object) in an image. In OpenCV, contours are typically extracted from a **binary image**, where pixels are split into **foreground** and **background**.

- For better accuracy, use binary images. So before finding contours, apply threshold or canny  edge detection
- The algorithm requires background to be white and borders to be black.
- `cv2.RETR_TREE` is contour retrieval mode
- `CHAIN_APPROX_SIMPLE` is the contour approximation method

```python
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
```

- Args
 	- **Retrieval mode** (`cv2.RETR_TREE`, `cv2.RETR_EXTERNAL`, …)
  		- `RETR_TREE`: returns _all_ contours and reconstructs full parent/child hierarchy (outer contours + holes).
  		- `RETR_EXTERNAL`: returns only the outermost contours (often simplest).
 	- **Approximation method** (`cv2.CHAIN_APPROX_SIMPLE`, `cv2.CHAIN_APPROX_NONE`)
  		- `CHAIN_APPROX_SIMPLE`: compresses horizontal/vertical segments (fewer points).
  		- `CHAIN_APPROX_NONE`: keeps every boundary pixel (more points).
- Returns:
 	- contours is a list`[np.array(x, y), ...]` of boundary points in the image. Shape is typically `(N, 1, 2)` and points are `(x, y)` (x = column, y = row).
 	- `hierarchy`: array describing contour nesting (parent/child relationships), useful for holes.

### 1) Input image (grayscale)

The raw image before any preprocessing. At this stage, contours are not well-defined because foreground and background are not separated yet.

![Input image](https://i.postimg.cc/HLrsNVFK/1-input.png)

### 2) Thresholded binary image

We convert the image into a **binary** representation so that the object pixels become one value (e.g., 255) and the background becomes 0 (or vice versa). Contour algorithms usually assume a clean binary separation.

![Thresholded image](https://i.postimg.cc/g0xk7nfb/2-threshold.png)

### 3) Foreground mask

We convert the binary image into a boolean mask (`foreground = binary != 0`), where `True` indicates object pixels. This makes logical operations (AND/OR/NOT) easy and fast.

![Foreground mask](https://i.postimg.cc/Gp4hfHZ0/3-foreground.png)

### 4) Boundary mask

We identify **boundary pixels** as foreground pixels that are _not fully surrounded_ by foreground in the 4-neighborhood. This produces a thin “outline” that is ideal for contour tracing.

![Boundary mask](https://i.postimg.cc/Gp4hfHZ1/4-boundary.png)

### Pseudocode

```
1. img = threshold(image) 
2. find_contours()
 1. foreground = (img != 0) # set black rgb=0 to true so black is background 
 2. boundary_img = foreground AND NOT(interior_4)
   up = move_up(boundary_image, 1_pixel)
   interior_4 = up AND down AND left AND right
   
 3. visited_boundary = zeros_like(boundary_img, false)
    
 4. neighbor_directions = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
        
 5. trace_single_contour(start_row, start_col):
     contour = [(start_row,start_col)]
     visited[start] = True
        current = start
        prev_dir_index = 0   # assume we came from "east" initially
        
     for _ in range(max_steps):
      for direction_idx in range(8):
       point_coord = [current_row, current_col] + neighbor_directions[direction_idx]
       if (boundary[point_coord] == 1):
       next_row, next_col = point_coord
      next_row, next_col  = current_row, current_col
      contour_pixels.append([current_row, current_col])
     return contour_pixels
   1. for each pixel (row,col) in boundary_img:
        if pixel == True and visited[row,col] == False:
            contours.append(trace_single_contour(row,col))

   h) unpad coordinates and return contours
```

- OpenCV can return **different starting points**
- OpenCV uses a specific **Suzuki-Abe algorithm**, can include holes/hierarchy
