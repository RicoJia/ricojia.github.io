---
layout: post
title: "[ML] D-PCC Encoder-Layers"
date: 2026-02-03 13:19
subtitle: sub-pixel convolution
header-img: img/post-bg-o.jpg
tags:
  - Machine-Learning
comments: true
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

### Downsampling Block

- FPS
- collapsed set: C(p): points whose nerest neighbor is P and get downsampled out.
- downsampling factor?

-----> Downsampled Point cloud

### Embeddings

- density embedding -> d-dimensional embedding MLP
- local position embedding (attention):
  - for each point in collapsed set
    - calculate direction and distance [pk - p] -> vec4[direction, distance
- ancestor embedding: point transformer layer to aggregate the previous stage feature (TODO?) of the collapsed points set -> representative sampled point (???) it's Uxd???
- all 3 embeddings -> MLP embedding -> vecd[ next stage]
-> local position embedding inflace [u, d] -> Attention

**IMPORTANT QUESTION: why this arrangement??**

### Entropy Encoding

- what was bottleneck in ML again?
- ps is quantized, how?
- Entropy Encoder
  - Need **arithmetic encoder** into the training process
  - How does the arithmetic encoder jointly optimize the entropy of the features?
- rate loss func?
