---
layout: post
title: "[ML] Data Formats"
date: 2026-01-23 13:19
subtitle: Pickle, pth, onnx
comments: true
header-img: img/post-bg-alibaba.jpg
tags:
  - ML
---

## `Pickle`

A Python serialization format for saving objects to disk and loading them back later. Common use cases include models, dictionaries, lists, pandas DataFrames, preprocessing objects, and intermediate results. It is Python-specific and not human-readable.

```python
import pickle

data = {"name": "Rico", "value": 42}

# Save to .pkl
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)

# Load from .pkl
with open("data.pkl", "rb") as f:
    loaded_data = pickle.load(f)

print(loaded_data)
```

## `.pt` / `.pth`

PyTorch's checkpoint format for storing model weights. You can save weights only:

```python
import torch

# Save weights only
torch.save(model.state_dict(), "model.pt")

# Load weights only
model.load_state_dict(torch.load("model.pt"))
model.eval()
```

Or a full checkpoint:

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "loss": loss,
}, "checkpoint.pt")
```

- `model_state_dict`: all learned weights and biases, stored as `{layer_name: tensor}`.
- `optimizer_state_dict`: internal optimizer state (momentum buffers, moving averages, learning rate, step counters), enabling exact resumption of training. Stored as:

```python
{
    "state": {...},
    "param_groups": [...]
}
```

- `epoch` and `loss` are stored as an integer and a float, respectively.

To resume training, load the checkpoint and restore each component:

```python
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

## `.onnx` (Open Neural Network Exchange)

A portable neural network format designed for deployment outside of training frameworks. It supports PyTorch, TensorFlow, ONNX Runtime, and more, with fast, hardware-optimized, cross-platform inference.

Export from PyTorch:

```python
torch.onnx.export(model, dummy_input, "model.onnx")
```

Run inference:

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"input": x})
```

ONNX is commonly used for C++ deployment on edge devices. One such graph typically includes:

| Component           | Meaning                               |
| ------------------- | ------------------------------------- |
| **Nodes**           | Operations (Conv, MatMul, ReLU, etc.) |
| **Edges / tensors** | Data flowing between nodes            |
| **Inputs**          | Model inputs                          |
| **Outputs**         | Model outputs                         |
| **Initializers**    | Learned weights (parameters)          |

For example:

```python
y = ReLU(Wx + b)
```

becomes

```
Input (x)
   ↓
MatMul (W * x)
   ↓
Add (+ b)
   ↓
ReLU
   ↓
Output (y)
```

Each box is a node in the graph. Graph engines can optimize the graph. E.g., `Conv → BatchNorm → ReLU` may become `FusedConv` which is much faster. ️Also, a graph lets runtime detect independent operations and run them in Parallel.

A graph lets runtimes detect independent operations and run them in parallel.

### ONNX Graph

An ONNX graph represents the neural network as a computational graph. Connections are connected together to show how data flows from input -> output.

### Caution: Not All Components Are ONNX-Exportable

Not every part of a model can be traced into an ONNX graph. The table below summarizes exportability for a typical point-cloud compression model:

| Component | ONNX-exportable? | Reason |
|---|---|---|
| `model.decoder` | ✅ Yes | Pure `Conv1d`/`Conv2d`/reshape — traced cleanly |
| `model.pre_conv` | ✅ Yes | `Conv1d` + `GroupNorm` + `ReLU` |
| `model.latent_xyzs_synthesis` | ✅ Yes | `Conv1d` stack |
| `model.encoder` | ❌ No | Uses `pointops.furthestsampling` + `knnquery_heap` — CUDA custom ops, not traceable |
| `model.feats_eblock.compress/decompress` | ❌ No | `compressai` range coder — pure Python entropy coding, not a torch graph |
| `model.feats_eblock.forward` | ⚠️ With mock | Can be traced if `__round__` is mocked, as `visualize_model.py` already does |

**Hard blocker — the entropy coder.** `EntropyBottleneck.compress()` / `.decompress()` are Python-level range coders that produce byte strings, not tensors. There is no torch graph to export, so they cannot be represented in ONNX.

The realistic deployment split is to keep the encoder and entropy coding in Python/CUDA, and export only the decoder side to ONNX:

```
┌─ encode (Python/CUDA) ────────────────────────────┐
│  encoder → latent_xyzs, latent_feats              │
│  feats_eblock.compress() → byte strings           │
└───────────────────────────────────────────────────┘

┌─ decode (ONNX-exportable) ────────────────────────┐
│  feats_eblock.decompress() → latent_feats         │
│  decoder → reconstructed output                  │
└───────────────────────────────────────────────────┘
```

**Q: Could a C++ reimplementation of `compress()` be ONNX-exported?**
It doesn't need to be. If `compress()` / `decompress()` are reimplemented natively in C++, they are called directly from C++ code — completely outside the ONNX graph. The ONNX model only needs to cover the neural network computations (i.e., the decoder). The entropy coder lives alongside it as a separate C++ component, not inside the graph.

**C++ decoder-side skeleton:**

```cpp
#include <onnxruntime_cxx_api.h>
#include <vector>

// ── 1. Load ONNX decoder ────────────────────────────────────────────────────
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "decoder");
Ort::SessionOptions opts;
opts.SetIntraOpNumThreads(1);
Ort::Session session(env, "decoder.onnx", opts);

// ── 2. Entropy decode (C++ range coder) ─────────────────────────────────────
// byte_strings + CDF params (loaded from a .json or .bin sidecar) → latent_feats
// e.g. using a C++ arithmetic coding library or a hand-ported compressai coder
std::vector<float> latent_feats = entropy_decode(byte_strings, cdf_params);
std::vector<float> latent_xyzs  = decode_xyzs(xyz_byte_strings);

// ── 3. Run ONNX decoder ──────────────────────────────────────────────────────
auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

std::array<int64_t, 3> feats_shape{1, C, N};
std::array<int64_t, 3> xyzs_shape{1, 3, N};

Ort::Value inputs[] = {
    Ort::Value::CreateTensor<float>(memory_info,
        latent_feats.data(), latent_feats.size(),
        feats_shape.data(), feats_shape.size()),
    Ort::Value::CreateTensor<float>(memory_info,
        latent_xyzs.data(), latent_xyzs.size(),
        xyzs_shape.data(), xyzs_shape.size()),
};

const char* input_names[]  = {"latent_feats", "latent_xyzs"};
const char* output_names[] = {"reconstructed"};

auto outputs = session.Run(Ort::RunOptions{},
    input_names, inputs, 2,
    output_names, 1);

float* reconstructed = outputs[0].GetTensorMutableData<float>();
```

The CDF parameters learned during training (used by the entropy coder) are not part of the ONNX graph — they are exported separately (e.g., as a `.json` or `.bin` sidecar file) and loaded by the C++ entropy decoder at runtime.

## `.json`

Used for model metadata, label mappings, experiment configurations, and similar structured data.
