---
layout: post
title: "[ML] D-PCC Losses"
date: 2026-02-11 13:19
subtitle: Entropy Bottleneck
header-img: img/post-bg-o.jpg
tags:
  - Machine-Learning
comments: true
---
---

## Entropy Encoding

### What is the bottleneck?

In learned compression the **entropy bottleneck** is a learned probability model $p(\hat{z})$ placed on the quantized latent features $\hat{z}$. It serves two purposes:

- **Training**: adds a rate penalty $R = -\sum \log_2 p(\hat{z})$ to the loss, pushing the encoder to produce features that are cheap to code.
- **Inference**: provides the CDF needed by the range coder to compress $\hat{z}$ into a bitstream.

### How are the features quantized?

During training, hard rounding is replaced by additive uniform noise to keep gradients flowing:

$$\tilde{z} = z + u, \qquad u \sim \mathcal{U}(-0.5,\, 0.5)$$

At inference, the features are hard-rounded to integers: $\hat{z} = \text{round}(z)$.

### How does the arithmetic/range coder fit into training?

It does **not** run during training. Instead, the expected bit cost is computed analytically using the learned CDF $F$:

$$R = -\sum_i \log_2 \bigl[F(\hat{z}_i + 0.5) - F(\hat{z}_i - 0.5)\bigr]$$

This is differentiable, so it can be backpropagated. The range coder is only called at inference time via `EntropyBottleneck.compress()` / `decompress()`.

### Rate–distortion loss

$$\mathcal{L} = D + \lambda R$$

- $D$ — reconstruction distortion (e.g. Chamfer distance between original and reconstructed point cloud).
- $R$ — estimated bit-rate from the entropy bottleneck (bits per point).
- $\lambda$ — trade-off weight: larger $\lambda$ → smaller files at the cost of more distortion.

The joint optimisation drives the encoder to produce a **compact, peaked latent distribution** (low $R$) while still enabling accurate reconstruction (low $D$).

---

## Loss

```python
Chamfer Loss (pre_downsample_points_per_stage, decoded_points_per_stage):
 for i in layers:
  ground_truth = pre_downsample_points_per_stage[layer_num-i-1]
   decoded_prediction = decoded_points_per_stage[i]
  d1, d2, _, pred2gt_idx = chamfer(ground_truth, pred)
   raw_chamfer_loss = d1.mean() + d2.mean()
        loss = loss + (raw_chamfer_loss if i == layer_num - 1 else (chamfer_coe * raw_chamfer_loss))
```

get_density_loss

```python
get_density_loss(
    gt_dnums, gt_mdis: # downsampling numbers and mean distance from orig point cloud to downsampled cloud, from encoder stages of (B, N_gt_s) tensors
 pred_unums, pred_mdis: # predictions of upsampling numbers and mean distance from compressed point to its upsampled points, from decoder stages of (B, N_pred_i) tensors
    all_pred2gt_idx: #  decoder stages of (B, N_pred_i) mapping pred->gt
):
    l1 = nn.L1Loss(reduction="mean")
    loss = gt_dnums[0].new_zeros(())
 for i in range(layer_num):
  
```

 mean distance, number of upsampled points, Chamfer loss per downsample stage is fed into the loss function, so they are directly penalized:

- Upsampling ratioL: L1(predicted upsample_num  vs  ground truth downsample_num)
- predicted mean distance  vs  true mean distance

# TODO: generate description

```
def get_pts_num_loss(
    gt_xyzs: List[torch.Tensor],
    pred_unums: List[torch.Tensor],
    layer_num: int,
    pts_num_coe: float,
) -> torch.Tensor:
    """Penalises the total predicted point count diverging from the ground-truth count.

    For each decoder stage ``i``, sums all predicted upsample counts and
    compares to the total number of ground-truth points at the corresponding
    encoder stage. The loss is the absolute difference, summed across stages.

    Args:
        gt_xyzs: ``[layer_num]`` tensors ``(B, 3, N_gt_s)`` — encoder point clouds.
        pred_unums: ``[layer_num]`` tensors ``(B, N_pred_i)`` — predicted upsample counts.
        layer_num: number of stages.
        pts_num_coe: global scale applied to the total loss.
    """
    _assert_list_len("gt_xyzs", gt_xyzs, layer_num)
    _assert_list_len("pred_unums", pred_unums, layer_num)

    B = gt_xyzs[0].shape[0]
    loss = gt_xyzs[0].new_zeros(())

    for i in range(layer_num):
        ref_xyzs = gt_xyzs[layer_num - 1 - i]
        target_total = float(ref_xyzs.shape[2] * B)
        loss = loss + torch.abs(pred_unums[i].sum() - target_total)

    return loss * pts_num_coe
```

The final loss:

    total = chamfer + dens + pnum
