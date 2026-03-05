---
layout: post
title: "[ML] Model Visualization"
date: 2026-02-26 13:19
subtitle: 
header-img: img/post-bg-o.jpg
tags:
  - Machine-Learning
comments: true
---

## Netron

[Netron](https://netron.app) is a viewer for neural network models. The typical workflow is to export a PyTorch model to ONNX format and then drag the `.onnx` file into the Netron web app.

**What is ONNX?** Open Neural Network Exchange Format — a standard representation for ML models that is also faster to run than native PyTorch inference.

### Exporting to ONNX

Wrap the model to control its output signature, then call `torch.onnx.export`:

```python
class AutoEncoderONNXWrapper(nn.Module):
    """Returns only the reconstructed xyzs tensor (B, 3, N_out)."""

    def __init__(self, model: AutoEncoder):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decompressed_xyzs, _loss, _loss_items, _bpp = self.model(x)
        return decompressed_xyzs


wrapper = AutoEncoderONNXWrapper(model)
wrapper.eval()

with torch.no_grad():
    torch.onnx.export(
        wrapper,
        dummy_input,
        out_path,
        input_names=["point_cloud"],          # (B, C, N)
        output_names=["reconstructed_xyzs"],   # (B, 3, N_out)
        dynamic_axes={
            "point_cloud": {0: "batch", 2: "num_points"},
            "reconstructed_xyzs": {0: "batch", 2: "num_points_out"},
        },
        opset_version=17,
        verbose=False,
    )
```

Then visualize:

```bash
python visualize_model.py --onnx-only
# then drag model.onnx to https://netron.app
```

### ONNX Node Reference for Mock Modules

These are ONNX lowerings of specific Python lines in the mock modules. Each maps 1-to-1:

**`Slice`** — from `.narrow(2, 0, M)` in the encoder:

```python
cur_feats = layer(cur_feats).narrow(2, 0, M)   # keep first M points
cur_xyzs  = cur_xyzs.narrow(2, 0, M)
```

Mock FPS downsampling — slices `(B, C, 512) → (B, C, 170)` along the points dimension. The real encoder uses `pointops.furthestsampling` instead.

**`Tile`** — from `.repeat(1, 1, factor)` in the decoder:

```python
cur_feats = layer(cur_feats).repeat(1, 1, factor)   # upsample ×8
cur_xyzs  = cur_xyzs.repeat(1, 1, factor)
```

Mock upsampling — copies each point `factor` times: `(B, C, 18) → (B, C, 144)`. The real decoder generates child point positions via learned MLPs instead.

**`ConstantOfShape + Expand`** — boilerplate from PyTorch's ONNX exporter to implement `.repeat()` when the repeat count is computed at runtime:

```
ConstantOfShape → Expand → Tile
```

`ConstantOfShape` creates a shape vector `[1, 1, factor]`, `Expand` broadcasts it to match the tensor rank, and `Tile` does the actual repetition. Together they form a single "upsample ×N" block.

## torchviz

pip install torchinfo torchviz
