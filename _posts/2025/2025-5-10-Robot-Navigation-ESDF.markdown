---
layout: post
title: Robot Navigation ESDF
date: 2025-5-1 013:19
subtitle:
header-img: img/post-bg-kuaidi.jpg
tags:
  - Robotics
  - SLAM
comments: true
---
## ESDF

There are two common approaches for incremental ESDF construction in robotics: **Voxblox** and **Fiesta**.  

Here we describe the core idea behind the **1D Euclidean Distance Transform (EDT)** used in [Fast Planner](https://github.com/HKUST-Aerial-Robotics/Fast-Planner),  based on: [_Felzenszwalb & Huttenlocher, “Distance Transforms of Sampled Functions”_](https://cs.brown.edu/people/pfelzens/papers/dt-final.pdf)

## 1D Euclidean Distance Transform

Consider a 1D voxel grid with indices

$$
p \in [0, n-1]
$$

- Distance is euclidean distance, not Manhattan Distance.
- f(p) = 0 at obstacle locations, otherwise it's  $\inf$

Assume obstacles are in $Q$, then the distance field value at index $p$ is  to find the minimum distance among all obstacles Q:

$$
D(p) = min_{Q} ((p - q) ^ 2 + f(q)
$$

The goal  is equivalent to finding the lower **Envelope** of the quadratic functions

![](https://i.postimg.cc/xTqJsbsY/Screenshot-from-2026-02-16-18-42-22.png)

It's easy to find the intersection between two such parabolas:

$$
(p-q_1)^2 + f(q_1) - ((p-q_2)^2 + f(q_2))

\\

\Rightarrow
\\
p = \frac{f(q_1) + q_1^2 - f(q_2) - q_2^2}{2(q_1 - q_2)}
$$

Then, voxels in to the left and the right of the intersections will get distance field values from the corresponding parabola.

Now we are going to have one very important property:
> Since all parabolas have the same quadratic terms, **any two parabolas intersect at most once despite of their relative positions**

This property is what makes the algorithm linear time.

So now, we can start finding distance value from the **leftmost** 1D voxel.

```python

# f is 1D array. Distance from x to parabola p is (x-p)^2 + f[p]
# obstacles at m will have f[m] = 0. Free cell n has f[n] = INF
def distance_tf_1d(f):

 # 
 n = f.shape[0] # number of voxels 
 v = np.zeros(n, dtype=np.int32)  # The x coordinate of the lowest point of each parabola, the "valleys"
 z = np.zeros(n + 1, dtype=np.float64) # left boundaries of all parabolas   
 d = np.zeros(n, dtype=np.float64) # final distance field  

 k = 0 
 v[0] = 0
 z[0] = - INF
 z[1] = + INF
 
 
 # intersection between two parabolas
 def intersect(q1, q2):
  return ((f[q1] + q1*q1) - (f[q2] + q2*q2)) / (2.0*(q1 - q2))

 for q in range(0,n):
  p = intersect(v[k], q)
  # q will only be larger than v[k]. If the intersection is to the LEFT of the k-th parabola's break point, that means q-th parabola is below k-th parabola for that entire range. In that case, we will replace the k-th parabola
  while p <= z[k]:
   k -= 1
   p = intersect(v[k], q)
  k += 1
  v[k] = q
  z[k] = p
  z[k+1] = INF

 # walk along x axis from 0 to n
 # you now have z[k+1] defines the break points of the envelope
 # v[k] is the index of parabolas at k.
 k = 0
 for q in range(n):
  while z[k+1] < q:
   k += 1
  p = v[k]
  d[q] = (q - p) * (q - p) + f[p]
 return d

```

## Extending to N-D

The beauty of this algorithm is that for N-dimensions, you can apply this algorithm independently.

Assume that we have a 2D map. First, apply 1D Distance Transform along each row (x axis). Then, for the rows with obstacles, each cell has a distance value to their nearest row obstacle.
$$
D(x,y) = min ((x - i) ^ 2 + F(i,y))
$$

Now, apply distance transform along each column.
$$
D(x,y) = min_j((y - j)^2 + G(x,j) )
$$
You scan through voxels along this column, which already provide distance values there. Still clear as mud? Let's walk through an example:

In this grid, let the **only obstacle be at (2,1)**.

```
(0,2) (1,2) (2,2)  
(0,1) (1,1) (2,1)  
(0,0) (1,0) (2,0)
```

So We want squared Euclidean distance:

$$
D(x,y) = (x - 2)^2 + (y-1)^2
$$

### Step 0 — Initial grid F

We define:

- F = 0 at obstacle
- F = INF elsewhere

```
y=2:  INF  INF  INF  
y=1:  INF  INF   0  
y=0:  INF  INF  INF
```

### Step 1 — 1D EDT along x (row by row)

We now compute for each row:

$$
D(x,y) = min ((x - i) ^ 2 + F(i,y))
$$

```
y=2:  INF  INF  INF  
y=1:    4    1    0  
y=0:  INF  INF  INF
```

### Step 2 — 1D EDT along y (column by column)

$$
D(x,y) = (x - 2)^2 + (y-1)^2
$$
Column 1 is

```
y=2: INF  
y=1: 4  
y=0: INF
```

For y=0, x=1,  

$$
D(0,y) = (y-1)^2 + 4 = 5
$$

Then you can see that 4 already the x-axis distance of the column to the obstacle. Adding $(y-1)^2$ directly gives the distance! So the final values are:

```
y=2:  5   2   1  
y=1:  4   1   0  
y=0:  5   2   1
```

Which is exactly $D(x,y) = (x - 2)^2 + (y-1)^2$

## Reference

[1]  [Zhihu](https://zhuanlan.zhihu.com/p/671385710
)
