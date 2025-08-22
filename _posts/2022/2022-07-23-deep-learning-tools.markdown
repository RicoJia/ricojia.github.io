---
layout: post
title: Deep Learning - Tools
date: '2022-07-23 13:19'
subtitle: wandb, tqdm
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Wandb

`wandb` is a visualization tool that records various deep learning experiment data. It never discloses what databases it uses, but it might be a combination of cloud-based scalable databases such as relational databases (PostgreSQL), non-relational databases (MongoDB, DynamoDB). Specifically, it keeps track of:

- Metrics: losses, accuracy, loss, precision, recall, etc.
- Model checkpoints: snapshot of model parameters during training for later retrieval and comparison
- Gradients and weights: can record changes in model weights during training.
- Images, Audio, Other media

How to get started? [Their page has a good introduction](https://github.com/wandb/wandb?tab=readme-ov-file)

One nice feature of `wandb` is that once you've set up your account and logged in on the training machine, you will get a link to your project and visualize **almost live** (you need to refresh the page though).

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/f135d44f-bea9-40ee-8279-78dc356cc77b" height="300" alt=""/>
    </figure>
</p>
</div>

My boiler plate is:

```python
import wandb
wandb_logger = wandb.init(
    project="Rico-mobilenetv2", resume="allow", anonymous="must"
)
wandb_logger.config.update(
    dict(
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE * ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        training_size=len(train_dataset),
        amp=USE_AMP,
        optimizer=str(optimizer),
    )
)
logging.info(
    f"""🚀 Starting training🚀 :
    Epochs:          {NUM_EPOCHS}
    Batch size:      {BATCH_SIZE}
    Learning rate:   {LEARNING_RATE}
    Weight decay:    {WEIGHT_DECAY}
    Training size:   {len(train_dataset)}
    Device:          {device.type}
    Mixed Precision: {USE_AMP},
    Optimizer:       {str(optimizer)}
"""
)
wandb.watch(model, log_freq=100)

model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    wandb_logger.log(
        {
            "epoch loss": epoch_loss,
            "epoch": epoch,
            "learning rate": current_lr,
            "total_weight_norm": total_weight_norm,
            "elapsed_time": timer.lapse_time(),
        }
    )

    images_t = ...  # generate or load images as PyTorch Tensors
    wandb.log({"examples": [wandb.Image(im) for im in images_t]})

wandb.finish()
```

- `wandb.watch(model, log_freq=100)` logs gradients and weights every 100 batches, when `log()` is called. [For more, see here](https://docs.wandb.ai/guides/integrations/pytorch/#logging-gradients-with-wandb-watch)
- Wandb can log images as well. [For more, see here](https://docs.wandb.ai/guides/integrations/pytorch/#logging-images-and-media)

## tqdm

`tqdm` creates a progress bar for iterables. Here, I have an example:

```python
from tqdm import tqdm

with tqdm(total=image_num, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar: 
    pbar.update(inputs.size(0))  # Increment progress bar by number of images in the batch
    ...
    pbar.set_postfix(**{'loss (batch)': loss.item()})
```

unit is 'img' so we can see 'img/s' at the progress bar.
You should be able to see a progress bar:

```
Epoch 1/10:  |███████████-------| 600/1000 [00:30<00:15, 25.00img/s, loss (batch)=0.542]
```

Or, we can use `tqdm(iterable) -> iterable` and do not need to manually update it.

```python
word_to_vec_map_unit_vectors = {
    word: embedding / np.linalg.norm(embedding)
    for word, embedding in tqdm(word_to_vec_map.items())
}
```

- Binary bytes
  - `KiB`: kibibyte = 1024 bytes, `MiB`: Mebibyte = 1024 KiB,
  - `GiB` = 1024 MiB`PiB`, `TiB`: 1024 GiB, Pebibyte = 1024 TiB

### tqdm For Multiple Processes

tqdm bars by default only work with a single process. If we want to spawn multiple processes, one universal set up could be:

```python
from multiprocessing import Manager, Process

def worker(mcap_file: str, prog, total_msg: int):
    with open(mcap_file, "rb") as f:
        print(f"Processing {mcap_file}")
        for msg in get_msg():
            process_msg(msg)
            count += 1
            if count % 1000 == 0:
                prog[mcap_file] = count
        prog[mcap_file] = count

with Manager() as manager:
    mcap_paths = ["data1.mcap", "data2.mcap"]
    prog = manager.dict()
    total_msgs = {}
    for mcap_file in mcap_paths:
        total_msgs[mcap_file] = get_num_msgs(mcap_file)
    procs = []
    for mcap_file in mcap_paths:
        prog[mcap_file] = 0
        proc = Process(
            target=worker, args=(mcap_file, prog, total_msgs[mcap_file])
        )
        proc.start()
        procs.append(proc)

    bars = {
        mcap_file: tqdm(
            total=total_msgs[mcap_file],
            position=i,
            leave=True,
        )
        for (i, mcap_file) in enumerate(mcap_paths)
    }

    try:
        alive = True
        while alive:
            alive = any(proc.is_alive() for proc in procs)
            for p in mcap_paths:
                new_count = prog.get(p, bars[p].n)
                if new_count > bars[p].n:
                    bars[p].update(new_count - bars[p].n)
            sleep(0.5)
    finally:
        for pr in procs:
            pr.join()
        for bar in bars.values():
            bar.close()

```

## FiftyOne

- Running inferencing on GCP, $0.05/image,
  - Grouding dino (object detection & language prompts) 2im/s
  - Segment-Anything, 1 im/s
  - Post-processing non-maxima, non-singular suppresion
- Fiftyone supports vector db:

- Data augmentation with night, snow, and rain
  - How?
- Data is far superior than models. Faster-RCNN (2015), a toy model, trained on 100M images
  - no temporal tracking
