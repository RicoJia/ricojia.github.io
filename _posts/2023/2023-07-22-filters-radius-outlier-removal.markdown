---
layout: post
title: Robotics - [Filters] Radius Outlier Removal
date: '2023-07-22 13:19'
subtitle: Removing Points With Too Few Neighbors
comments: true
header-img: "img/post-bg-infinity.jpg"
tags:
    - Robotics
---

Radius outlier removal keeps a point only if it has at least `k` neighbors within radius `r`. Isolated points, such as sensor noise or stray returns, get removed.

https://i.postimg.cc/GtvRmxKz/ror-filter.png

## Vanilla Approach (O(N^2))

For each point `p_i`:

1. Loop through every other point `p_j`.
2. Calculate the Euclidean distance `||p_i - p_j||`.
3. Count the neighbors within radius `r`:

$$
\begin{gather*}
n_i = \left|\{\, j \neq i : \|p_i - p_j\|_2 \le r \,\}\right|
\end{gather*}
$$

4. Remove `p_i` if `n_i < k`.

```cpp
for (int i = 0; i < N; ++i) {
    int count = 0;
    for (int j = 0; j < N; ++j) {
        if (i == j) continue;

        if ((points[i] - points[j]).norm() <= radius)
            ++count;
    }
    keep[i] = count >= min_neighbors;
}
```

- Every point is compared against every other point, so the cost is `N(N-1)`, or O(N^2) distance checks.

## Grid-Accelerated Approach (Approximately O(N) For Bounded Local Density)

1. Bucket the points into a spatial hash grid with cell size `radius * 1.001`. The cell is slightly wider than `radius` to leave a margin for float rounding.
2. For each point, search only its own cell and the 26 adjacent cells (3x3x3). Because each cell is at least `radius` wide, every neighbor within `radius` must lie in one of these 27 cells. If a neighbor isn't found there, it doesn't exist.
3. Check the actual distance for every candidate in those cells. The grid search covers a cube, but the outlier test is a sphere, so some candidates are farther than `radius`.
4. Process the points in parallel, and stop counting a point's neighbors once it reaches `min_neighbors`.

If each cell holds a bounded number of points, each point checks a constant number of candidates, so the total cost is about O(N).

PCL's `pcl::RadiusOutlierRemoval` gives the same result, but it searches with a KD-tree (or an organized-neighbor search for organized clouds) instead of a hash grid. The function below uses PCL for `pcl::PointXYZI` and falls back to the hash grid for custom point types.

```cpp
void radius_outlier_removal(PointCloud& cloud, float radius, int min_neighbors) {
    const float r2 = radius * radius;
    const float cell_size = radius;

    // 1. Bucket points into a spatial hash grid
    HashGrid grid;
    for (size_t i = 0; i < cloud.size(); ++i)
        grid[cell_of(cloud[i], cell_size)].push_back(i);

    // 2. Check each point in parallel
    std::vector<uint8_t> keep(cloud.size(), false);
    std::vector<size_t> indices(cloud.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par_unseq,
                  indices.begin(), indices.end(), [&](size_t i) {
        int count = 0;

        // Search own cell + 26 adjacent cells
        for (const auto& cell : neighboring_cells(cloud[i], cell_size)) {
            for (size_t j : grid.get(cell)) {
                if (i == j) continue;

                if (squared_distance(cloud[i], cloud[j]) <= r2)
                    ++count;

                if (count >= min_neighbors) break;
            }
            if (count >= min_neighbors) break;
        }

        keep[i] = (count >= min_neighbors);
    });

    // 3. Retain points with sufficient neighbors
    cloud = filter_by_mask(cloud, keep);
}

```

- The hash key can collide, so each bucket stores a list of cells, and the search compares cell coordinates before using a cell's points.
- Comparing squared distances against `r2` skips the `sqrt` that `.norm()` does in the vanilla loop.
- The parallel loop only reads `grid` and `coords`, and each iteration writes only its own `keep[i]`, so it needs no locks.
