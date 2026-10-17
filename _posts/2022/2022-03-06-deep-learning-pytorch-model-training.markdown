---
layout: post
title: Deep Learning - PyTorch Model Training
date: '2022-03-06 13:19'
subtitle: Checkpointing, Op Determinisim,  🤗 HuggingFace Trainer
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Checkpointing

Checkpointing is a technique to trade compute for memory during training. Instead of storing all intermediate activations (outputs layers) for backprop, which consumes a lot of memory, checkpointing discards some and recomputes them during the backward pass.  Thus, this saves memory at the expense of additional computation

```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.inc = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Example layer
        self.inc = checkpoint.checkpoint(self.inc)  # Enable checkpointing

    def forward(self, x):
        x = self.inc(x)  # Checkpointed layer
        return x
```

checkpointing can be used on functions as well.

A full training checkpoint may also include weights, optimizer state, epoch, scheduler, etc. PyTorch recommends saving more than just the model state when resuming training.

## Seeding

With seeding, we want to reproduce the same:

| What | Example | Controlled by |
|---|---|---|
| Initial weights | Random init of conv / linear layers | `torch.manual_seed()` before building the model |
| Random operations in data loading | Selecting random seabed crops; subsampling points when a cloud exceeds 1,024 | `random` / `np.random` / `torch` seeds **inside each DataLoader worker** (`worker_init_fn`) |
| Order of training samples | Which samples land in each batch with `shuffle=True` | The DataLoader's `generator` |

The data-loading row is the easy one to miss: with `num_workers > 0`, each worker is a separate process, so seeding the main process alone does not make NumPy or Python's `random` reproducible inside the workers. The pattern from the [PyTorch reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html):

```python
def seed_worker(worker_id):
    # torch already gives each worker a distinct seed derived from the base seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4,
                    worker_init_fn=seed_worker, generator=g)
```

Seeding makes the random numbers repeat; making the GPU math itself repeat is the next section.

## Op Determinisim

Here is [a good reference on Op Determinism](https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/config/experimental/enable_op_determinism). Below is how this story goes

- Tensor operations are not necessarily deterministic:
  - `tf.nn.softmax_cross_entropy_with_logits` (From a quick search, it's still not clear to me why this is non-deterministic. Mathematically, the quantity should be deterministic.)
- Op Determinisim will make sure you get the same output with the same code, same hardware. But it will disable asynchronicity, so **it will slow down these operations**
  - Use the same software environment in every run (OS, checkpoints, version of CUDA and TensorFlow, environmental variables, etc). Note that determinism is not guaranteed across different versions of TensorFlow.
- How to enable Op Determinism?
  - PyTorch
  - TensorFlow:
        ```python
        tf.keras.utils.set_random_seed(1)
        tf.config.experimental.enable_op_determinism()
        ```

    - This effectively sets the pseudorandom number generators (PRNGs) in  Python seed, the NumPy seed, and the TensorFlow seed.
    - Without setting the seed, `tf.random.normal` would raise `RuntimeError`, but Python and Numpy won't

- Have more consistent training loss, one can do

```python
# Setup random seed
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

# Optional to enforce sequential execution
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False
```

## Data Provenance

In ML, **provenance** means the history of where something came from and what happened to it. For data, provenance answers questions like:

| Question | Example record |
|---|---|
| Where was this sample collected? | Site, run name, GPS / map region |
| Who or what labeled it? | Annotator ID, or the auto-labeling model and its version |
| What sensor produced it? | Sensor model, serial number, calibration file |
| When was it recorded? | Timestamp of the source bag / log |
| What preprocessing was applied? | Crop, resample, normalization, and the code commit that ran it |
| Which dataset split did it enter? | train / val / test, and the split seed or list file |
| Which model version trained on it? | Checkpoint name or training run ID |

Seeding and op determinism make a training run repeatable; provenance is what lets you trace a bad prediction back to the exact samples, labels, and preprocessing behind it, and check that no test sample leaked into training.

## HuggingFace Trainer

Hugging Face has a Trainer class that has distributed training by default, and mixed precision training in [a Trainer class](https://huggingface.co/docs/transformers/main_classes/trainer). It goes hand-in-hand with the TrainerArgument class

Boiler plate for image segmentation:

```python
def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    # Flatten the tensors
    preds = preds.flatten()
    labels = labels.flatten()
    # Compute IoU
    metric = load_metric("iou")
    iou = metric.compute(predictions=preds, references=labels, average="macro")
    return {"iou": iou}

training_args = TrainingArguments(
    output_dir="./segmentation_results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./segmentation_logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="iou",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
trainer.save_model("./trained_segmentation_model")
```
