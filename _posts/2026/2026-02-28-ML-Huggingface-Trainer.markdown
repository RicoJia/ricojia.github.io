---
layout: post
title: "[ML] HuggingFace Trainer"
date: 2026-02-26 13:19
subtitle:
header-img: img/post-infinity.jpg
tags:
  - Machine-Learning
comments: true
---

## Introduction

The [Trainer](https://huggingface.co/docs/transformers/v5.3.0/en/main_classes/trainer#transformers.Trainer) class provides a feature-complete training loop for PyTorch, supporting distributed training on multiple GPUs/TPUs and mixed precision via [NVIDIA Apex](https://nvidia.github.io/apex/), [AMD ROCm](https://rocm.docs.amd.com/en/latest/rocm.html), and [`torch.amp`](https://pytorch.org/docs/stable/amp.html). It pairs with [TrainingArguments](https://huggingface.co/docs/transformers/v5.3.0/en/main_classes/trainer#transformers.TrainingArguments) to control all training hyperparameters.

### Minimal Working Example

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments


# 1) Dataset
class PointCloudDataset(Dataset):
    def __init__(self, num_samples=1000, num_points=1024):
        self.num_samples = num_samples
        self.num_points = num_points

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        points = torch.randn(self.num_points, 3, dtype=torch.float32)  # [N, 3]
        label = torch.tensor(idx % 2, dtype=torch.long)
        return {"points": points, "labels": label}

train_dataset = PointCloudDataset(num_samples=500)
eval_dataset  = PointCloudDataset(num_samples=100)


# 2) Model
#    forward() must return a dict with at least "loss" and "logits".
#    Trainer reads "loss" for backprop; "logits" are forwarded to compute_metrics.
class SimplePointNet(nn.Module):
    def __init__(self, hidden_dim=64, num_classes=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, points=None, labels=None):
        x = self.mlp(points).mean(dim=1)   # [B, N, H] -> [B, H]
        logits = self.classifier(x)         # [B, C]
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

model = SimplePointNet()


# 3) Collator
#    The default collator pads 1-D sequences (designed for NLP tokens).
#    For non-standard tensor shapes like [N, 3] point clouds it fails,
#    so we provide a custom stack-based collator instead.
def collate_fn(batch):
    points = torch.stack([item["points"] for item in batch])  # [B, N, 3]
    labels = torch.stack([item["labels"] for item in batch])  # [B]
    return {"points": points, "labels": labels}


# 4) Metrics
#    compute_metrics receives an EvalPrediction(predictions=logits, label_ids=labels).
#    Both are numpy arrays at this point (Trainer calls .numpy() internally).
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": float((preds == labels).mean())}


# 5) Training arguments
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=1e-3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    fp16=True,                    # mixed precision (use bf16=True on Ampere+)
    remove_unused_columns=False,  # keep custom keys like "points"
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
```

### How Trainer uses `loss` and `logits`

`Trainer` calls `model.forward(**batch)` on every step. It then:

1. **Training** — reads `output["loss"]`, calls `.backward()`, and steps the optimizer. Your model computes the loss internally so `Trainer` never needs to know the loss function.
2. **Evaluation** — collects `output["logits"]` across the eval set, assembles them into a single numpy array, and passes `(logits, labels)` to `compute_metrics`.

### What is a collator?

Collate /ˈkōˌlāt/: `collect and combine texts, information, sets of figures in proper order`

A **collator** is the function that the `DataLoader` calls to merge a list of `__getitem__` results (one per sample) into a single batched tensor. The default `DataCollatorWithPadding` was designed for NLP: it pads 1-D token ID sequences to the same length. It does not know how to stack higher-dimensional tensors like `[N, 3]` point clouds, and it silently drops keys it does not recognise. The custom `collate_fn` above does a plain `torch.stack` and preserves all keys.

The flag `remove_unused_columns=False` is also required: without it, `Trainer` inspects the model's `forward()` signature and silently removes any batch key that does not appear as a parameter name.

### Mixed precision (fp16 / bf16)

Set `fp16=True` in `TrainingArguments`. `Trainer` wraps the forward/backward pass with `torch.autocast` and a `GradScaler` automatically — no manual model changes needed.

- **fp16** works on all CUDA GPUs with Tensor Cores (Volta+).
- **bf16** has a wider dynamic range and needs no grad scaling; requires Ampere+ (A100, RTX 30xx+) or TPUs.

---

## Constraints

### One loss, one optimizer

`Trainer` expects a single scalar `loss` from `forward()` and manages one optimizer internally. This becomes a problem when a model requires two separate optimizers — common in learned entropy coding (e.g., CompressAI).

**Example: point-cloud compression with an auxiliary entropy loss**

Entropy coding models learn a parametric cumulative distribution function (CDF) to estimate the probability of each quantised latent code. The CDF is parameterised by `.quantiles` — learnable values that define the median and tails of the distribution. These are trained by a separate **auxiliary loss** (a soft quantile regression loss), not by the main reconstruction loss (Chamfer distance, etc.), because:

- Their optimal learning rate and gradient scale differ from the main network weights.
- Mixing the gradients would corrupt the entropy estimates used during arithmetic coding.

With plain PyTorch you use two optimizers:

```python
main_params = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
aux_params  = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]

optimizer     = optim.Adam(main_params, lr=args.lr, weight_decay=args.weight_decay)
aux_optimizer = optim.Adam(aux_params,  lr=args.aux_lr)
scheduler     = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)

def train_step(batch):
    # Main backward (reconstruction + rate loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Auxiliary backward (entropy CDF quantiles)
    aux_optimizer.zero_grad()
    aux_loss.backward()
    aux_optimizer.step()
```

Because `Trainer` only manages one optimizer, this pattern requires subclassing `Trainer` and overriding `training_step()`, or bypassing `Trainer` entirely.

### Custom CUDA ops

Custom CUDA extensions (loaded via `torch.utils.cpp_extension` or `torch.ops`) work fine inside `forward()` — `Trainer` is agnostic to what happens inside the forward pass as long as the op returns standard `torch.Tensor` objects.

```python
import torch
from torch.utils.cpp_extension import load

# Build and load the extension once at import time
my_op = load(
    name="my_cuda_op",
    sources=["my_op.cu", "my_op_bind.cpp"],
    verbose=False,
)

class MyModel(nn.Module):
    def forward(self, points=None, labels=None):
        features = my_op.extract_features(points)  # returns a torch.Tensor
        logits = self.head(features)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}
```

If the op requires a hand-written CUDA backward kernel, implement it as a `torch.autograd.Function` with a `@staticmethod backward()`. `Trainer`'s fp16 `GradScaler` will still work correctly as long as the function participates in the autograd graph normally.
