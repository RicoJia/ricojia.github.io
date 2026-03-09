---
layout: post
title: "[ML] D-PCC Losses"
date: 2026-02-11 13:19
subtitle: Chamfer Loss, Density Loss, Point-Count Loss, Rate-Distortion Loss
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine-Learning
---

## Entropy Encoding

### What is the bottleneck?

In learned compression the **entropy bottleneck** is a learned probability model $p(\hat{z})$ placed on the quantized latent features $\hat{z}$. It serves two purposes:

- **Training**: adds a rate penalty $R = -\sum \log_2 p(\hat{z})$ to the loss, pushing the encoder to produce features that are cheap to code.
- **Inference**: provides the CDF needed by the range coder to compress $\hat{z}$ into a bitstream.

### How are the features quantized?

During training, hard rounding is replaced by additive uniform noise to keep gradients flowing:

$$\tilde{z} = z + u, \qquad u \sim \mathcal{U}(-0.5,\, 0.5)$$

At inference the features are hard-rounded to integers: $\hat{z} = \text{round}(z)$.

### How does the arithmetic/range coder fit into training?

It does **not** run during training. Instead, the expected bit cost is computed analytically using the learned CDF $F$:

$$R = -\sum_i \log_2 \bigl[F(\hat{z}_i + 0.5) - F(\hat{z}_i - 0.5)\bigr]$$

This is differentiable, so it can be back-propagated. The range coder is only called at inference time via `EntropyBottleneck.compress()` / `decompress()`.

### Rate-distortion loss

$$\mathcal{L} = D + \lambda R$$

- $D$ — reconstruction distortion (e.g. Chamfer distance between original and reconstructed point cloud).
- $R$ — estimated bit-rate from the entropy bottleneck (bits per point).
- $\lambda$ — trade-off weight: larger $\lambda$ → smaller files at the cost of more distortion.

The joint optimisation drives the encoder to produce a **compact, peaked latent distribution** (low $R$) while still enabling accurate reconstruction (low $D$).

---

## Loss Functions

The total training loss is the sum of three terms:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{chamfer}} + \mathcal{L}_{\text{density}} + \mathcal{L}_{\text{pts\_num}}$$

### Chamfer Loss

Measures geometric reconstruction quality at each decoder stage. At stage $i$, the predicted points are compared to the ground-truth point cloud from the **corresponding encoder stage** (i.e. the stage before downsampling destroyed detail). The final stage uses a full weight of 1; all earlier stages are scaled by `chamfer_coe < 1`.

```python
def get_chamfer_loss(
    pre_downsample_points_per_stage: List[torch.Tensor],  # (layer_num,) of (B, 3, N_gt_s)
    decoded_points_per_stage: List[torch.Tensor],         # (layer_num,) of (B, 3, N_pred_i)
    layer_num: int,
    chamfer_coe: float,
) -> torch.Tensor:
    loss = pre_downsample_points_per_stage[0].new_zeros(())
    for i in range(layer_num):
        ground_truth = pre_downsample_points_per_stage[layer_num - 1 - i]
        pred = decoded_points_per_stage[i]
        d1, d2, _, _ = chamfer(ground_truth, pred)
        raw = d1.mean() + d2.mean()
        loss = loss + (raw if i == layer_num - 1 else chamfer_coe * raw)
    return loss
```

#### Chamfer Loss Gradient Analysis

**Setup.** Take two 2D point sets:

$$P = \{p_1 = (0,0),\ p_2 = (2,0)\}, \qquad Q = \{q_1 = (0.5,0),\ q_2 = (3,0)\}$$

**Step 1 — Nearest-neighbor matching.**

- $P \to Q$: $p_1 \leftrightarrow q_1$,  $p_2 \leftrightarrow q_2$
- $Q \to P$: $q_1 \leftrightarrow p_1$,  $q_2 \leftrightarrow p_2$

**Step 2 — Loss value.**

$$L = \|p_1 - q_1\|^2 + \|p_2 - q_2\|^2 + \|q_1 - p_1\|^2 + \|q_2 - p_2\|^2 = 0.25 + 1 + 0.25 + 1 = 2.5$$

With the mean version ($|P|=|Q|=2$):

$$L_{\text{mean}} = \frac{0.25+1}{2} + \frac{0.25+1}{2} = 1.25$$

**Step 3 — Gradients.**

For a single squared term $\|p - q\|^2$:

$$\frac{\partial}{\partial p}\|p-q\|^2 = 2(p-q), \qquad \frac{\partial}{\partial q}\|p-q\|^2 = 2(q-p)$$

Each point appears in **two** symmetric terms (once from each direction), so its gradient doubles:

| Point | Contribution per term | Total gradient |
|---|---|---|
| $p_1 = (0,0)$ | $2(p_1 - q_1) = (-1, 0)$ | $\nabla_{p_1} L = (-2, 0)$ |
| $p_2 = (2,0)$ | $2(p_2 - q_2) = (-2, 0)$ | $\nabla_{p_2} L = (-4, 0)$ |
| $q_1 = (0.5,0)$ | $2(q_1 - p_1) = (1, 0)$ | $\nabla_{q_1} L = (2, 0)$ |
| $q_2 = (3,0)$ | $2(q_2 - p_2) = (2, 0)$ | $\nabla_{q_2} L = (4, 0)$ |

**Step 4 — Gradient descent effect** ($x \leftarrow x - \eta \nabla_x L$, $\eta = 0.1$).

Chamfer loss **pulls matched points toward each other**:

$$p_1: (0,0) \to (0.2,0) \qquad p_2: (2,0) \to (2.4,0)$$
$$q_1: (0.5,0) \to (0.3,0) \qquad q_2: (3,0) \to (2.6,0)$$

**One important Chamfer behavior — point clustering.**

A single point can become the nearest neighbor for multiple points simultaneously. For example:

$$P = \{(0,0),\ (0.2,0)\}, \qquad Q = \{(1,0)\}$$

Both $p_1$ and $p_2$ match $q_1$, so $q_1$ receives gradient from both:

$$\nabla_q L = 2(q - p_1) + 2(q - p_2)$$

$q$ is pulled toward the **average** of $p_1$ and $p_2$. This is one reason Chamfer loss can produce clustering: many predicted points can chase the same target, or one predicted point can represent several targets imperfectly.

> **Single-sided vs. bidirectional Chamfer.**
> The two directions play distinct roles:
>
> - $P \to Q$ (**accuracy**): every predicted point must be near some GT point.
> - $Q \to P$ (**coverage**): every GT point must be near some predicted point.
>
> With only $P \to Q$, the coverage term is absent. All predicted points can collapse onto a single GT point and still achieve low loss — there is no gradient to spread them over uncovered GT points. Single-sided loss therefore **worsens** the clustering / mode-collapse problem compared to the full bidirectional loss.

### Density Loss

Penalises two per-point density statistics that the decoder predicts for each upsampling stage:

1. **Upsampling count** — L1 between the predicted number of children per point (`pred_unums`) and the ground-truth downsampling count (`gt_dnums`).
2. **Mean distance** — L1 between the predicted mean child-to-parent distance (`pred_mdis`) and the ground-truth mean distance (`gt_mdis`).

For each decoder stage $i$, the nearest ground-truth point is looked up via `pred2gt_idx` so that predictions are compared to the correct GT entry.

```python
def get_density_loss(
    gt_dnums: List[torch.Tensor],        # (layer_num,) of (B, N_gt_s) — GT downsample counts
    gt_mdis: List[torch.Tensor],         # (layer_num,) of (B, N_gt_s) — GT mean distances
    pred_unums: List[torch.Tensor],      # (layer_num,) of (B, N_pred_i) — predicted upsample counts
    pred_mdis: List[torch.Tensor],       # (layer_num,) of (B, N_pred_i) — predicted mean distances
    all_pred2gt_idx: List[torch.Tensor], # (layer_num,) of (B, N_pred_i) — pred→GT index mapping
    layer_num: int,
    density_coe: float,
) -> torch.Tensor:
    l1 = nn.L1Loss(reduction="mean")
    loss = gt_dnums[0].new_zeros(())

    for i in range(layer_num):
        pred2gt = all_pred2gt_idx[i]                          # (B, N_pred_i)
        gt_stage = layer_num - 1 - i                         # matching encoder stage

        gt_dnum_matched = torch.gather(gt_dnums[gt_stage], 1, pred2gt)
        gt_mdis_matched = torch.gather(gt_mdis[gt_stage],  1, pred2gt)

        loss = loss + l1(pred_unums[i].float(), gt_dnum_matched.float())
        loss = loss + l1(pred_mdis[i],          gt_mdis_matched)

    return loss * density_coe
```

### Point-Count Loss

Penalises the **total** predicted point count at each decoder stage diverging from the total ground-truth point count at the corresponding encoder stage. This is a coarser global consistency check: it does not require point-to-point matching and simply checks that the decoder produces roughly the right number of points at each resolution.

```python
def get_pts_num_loss(
    gt_xyzs: List[torch.Tensor],      # (layer_num,) of (B, 3, N_gt_s) — encoder point clouds
    pred_unums: List[torch.Tensor],   # (layer_num,) of (B, N_pred_i)   — predicted upsample counts
    layer_num: int,
    pts_num_coe: float,
) -> torch.Tensor:
    loss = gt_xyzs[0].new_zeros(())

    for i in range(layer_num):
        ref_xyzs = gt_xyzs[layer_num - 1 - i]
        target_total = float(ref_xyzs.shape[2] * ref_xyzs.shape[0])  # N_gt_s * B
        loss = loss + torch.abs(pred_unums[i].sum() - target_total)

    return loss * pts_num_coe
```
