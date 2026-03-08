---
layout: post
title: "[ML] D-PCC Encoder-Layers"
date: 2026-02-03 13:19
subtitle: sub-pixel convolution
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine-Learning
---

[D-PCC paper](https://yunhe20.github.io/D-PCC/)

## Terminology

- **latent feature**: a compressed internal representation that contains essential information needed to reconstruct the original data.
- **Cardinality**: the number of elements in a set. If a point cloud P is: `P = {p1, p2, ..., pN}` cardinality  $C(s) = N$

---

## Encoder

A point cloud has no order, but it has geometric structure — planes, edges, etc. The PointTransformer and its position embedding learn the relative pose between each point and its K nearest neighbors.

In position embedding, direction and distance are encoded so a mapping into feature space is learned:

```python
rel       = knn_xyzs - q_xyzs.unsqueeze(-1)
direction = normalize(rel)
distance  = norm(rel)
```

In point transformers, self-attention without positional embedding depends only on feature content:

$$
\text{attn}(q, k, v) = \sum_k \alpha(q, k)\, v
$$

Injecting relative-position encoding makes attention geometry-aware — it can now differentiate a neighbor to the left from one above:

$$
d = \text{xyz} - \text{xyz}_{knn} \quad \text{(relative pose)}
$$

$$
\text{pos\_enc} = \text{MLP}(d)
$$

$$
\text{attn} = \text{MLP}(q - k + \text{pos\_enc})
$$

### MaskedPointTransformer

Masked attention over a KNN graph (neighbors are precomputed), where each query point attends to its K nearest key/value points. The mask can further restrict which neighbors are valid.

For each query point the layer:

1. Looks at its K neighbors in the key set.
2. Computes attention using feature differences and relative positions.
3. Ignores masked neighbors.
4. Returns a new feature vector per query point (with a residual connection).

**Inputs:**

- `q_xyz` `(B, 3, M)` — query points (often FPS-downsampled).
- `k_xyz` `(B, 3, N)` — key/value points (the original point set).
- `q_feat` `(B, C_in, M)` — feature vectors for query points, used for:
  - forming query vectors: `q = Wq(q_feat)`
  - the residual connection at the end
- `k_feat` `(B, C_in, N)` — features at key points; used to form `k = Wk(k_feat)`, then gathered into KNN neighborhoods.
- `v_feat` `(B, C_in, N)` — features at value points; used to form `v = Wv(v_feat)`, then gathered into KNN neighborhoods. Often the same as `k_feat`, but the signature allows them to differ.
- `knn_index` `(B, M, K)` — for each query point, the indices of its K nearest neighbors.
- `mask` `(B, M, K)` bool — which neighbors are valid.

### What the layer computes

1. **Gather neighborhood geometry** — each query gets a local patch:

    ```python
    knn_xyz = index_points(k_xyz, knn_idx)  # (B, 3, M, K)
    k = index_points(Wk(k_feat), knn_idx)   # (B, H, M, K)
    v = index_points(Wv(v_feat), knn_idx)   # (B, H, M, K)
    q = Wq(q_feat)                          # (B, H, M)
    ```

2. **Relative position encoding** — turns the spatial offset into a learned feature:

    ```python
    pos = delta(q_xyz.unsqueeze(-1) - knn_xyz)  # (B, H, M, K)
    ```

3. **Attention logits** — combine feature difference `q - k` with position `pos`:

    ```python
    attn = gamma(q.unsqueeze(-1) - k + pos)  # (B, H, M, K)
    ```

4. **Masked softmax** — invalid neighbors are set to $-\infty$ before softmax so they contribute zero probability. `mask[:, None]` broadcasts to `(B, 1, M, K)` across the H head dimension:

    ```python
    attn = attn.masked_fill(~mask[:, None], float('-inf'))
    attn = softmax(attn, dim=-1)
    ```

5. **Weighted aggregation** — sum neighbor values weighted by attention, adding position features to inject geometry into the aggregated representation:

    ```python
    agg = (attn * (v + pos)).sum(dim=-1)  # (B, H, M)
    ```

**Output:**

```python
return out(agg) + q_feat  # (B, C_out, M)
```

`out` is `Conv1d(hidden, C_out, 1)`. The channel size is kept constant so the residual `+ q_feat` is always valid. This produces an updated feature for each of the M query points.

### Embeddings

Each encoder stage collapses a set of input points into fewer representative points (via FPS/downsampling). For every surviving representative point, three embeddings are computed and then fused:

**1. Density embedding**

Encodes how many original points were collapsed into this representative. An MLP maps the count (or a soft density estimate) to a $d$-dimensional vector. This lets the decoder know how densely populated each region was so it can upsample by the right amount.

**2. Local position embedding**

For each collapsed point $p_k$ in the neighbourhood of representative $p$, the relative offset is computed:

$$\delta_k = p_k - p$$

Direction and distance are extracted from $\delta_k$ and passed through an MLP to produce a $d$-dimensional geometry descriptor. This captures the local surface shape around each representative.

**3. Ancestor embedding**

The features from the **previous encoder stage** (carried on the collapsed points) are aggregated into the representative via a `MaskedPointTransformer`. This propagates multi-scale context — details learned at finer scales are preserved as the point cloud is progressively compressed.

The three embeddings are concatenated and fused by another MLP to produce the final $d$-dimensional feature vector that travels to the next stage.

**Why separate coordinates and features?**

A point cloud is stored as two parallel tensors:

- **Coordinates** `(B, 3, N)` — where each point is in 3D space.
- **Features** `(B, C, N)` — what each point *means* in the context of compression.

Raw $(x,y,z)$ alone says only *where* a point is. Compression additionally needs to record *how densely packed* the original cloud was around each anchor, *what the local surface shape looks like*, and *what was learned at earlier encoder stages*. Three numbers per point cannot carry all of that information, which is why a $C$-dimensional feature vector is maintained alongside every point.

By the bottleneck those features encode:

- **Density** — how many original points collapsed into each anchor (density embedding).
- **Local geometry** — the spatial arrangement of those collapsed points (local position embedding).
- **Multi-scale context** — information propagated from the previous encoder stage (ancestor embedding).

Recall that during encoding, for each downsampled point p, downsampling discarded points collapse into it. This information is not losslessly transmitted but fused into the features. During decoding, in order to properly upsam- ple each point, we apply MLPs to predict an upsampling factor uˆ ≈ u from the features

This mirrors how learned **large image compression** works: a CNN encoder does not collapse an image to a single global vector; it produces a spatial feature map `(H/s × W/s × C)` so each decoder element has access to locally relevant information. D-PCC does the same on point clouds — the bottleneck is a sparse set of anchor points with feature vectors `(B, C, N_s)`.

The decoder then upsamples each anchor, using its feature vector to determine how many new points to generate and where to place them.

---

## Entropy Encoding

### What is the bottleneck?

The **entropy bottleneck** is a learned probability model $p(\hat{z})$ placed on the quantized latent features $\hat{z}$. It serves two purposes:

- **Training**: adds a rate penalty $R = -\sum \log_2 p(\hat{z})$ to the loss, pushing the encoder to produce features that are cheap to code.
- **Inference**: provides the CDF the range coder needs to compress $\hat{z}$ into a bitstream.

### How are the features quantized?

During training, hard rounding is replaced by additive uniform noise to keep gradients flowing:

$$\tilde{z} = z + u, \qquad u \sim \mathcal{U}(-0.5,\, 0.5)$$

At inference the features are hard-rounded: $\hat{z} = \text{round}(z)$.

### How does the range coder fit into training?

It does **not** run during training. The expected bit cost is computed analytically from the learned CDF $F$:

$$R = -\sum_i \log_2 \bigl[F(\hat{z}_i + 0.5) - F(\hat{z}_i - 0.5)\bigr]$$

This is differentiable, so it can be backpropagated. The range coder is only called at inference via `EntropyBottleneck.compress()` / `.decompress()`.

### Rate–distortion loss

$$\mathcal{L} = D + \lambda R$$

- $D$ — reconstruction distortion (e.g. Chamfer distance).
- $R$ — estimated bit-rate from the entropy bottleneck (bits per point).
- $\lambda$ — trade-off weight: larger $\lambda$ → smaller bitstream at the cost of higher distortion.
