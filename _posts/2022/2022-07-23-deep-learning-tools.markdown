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
experiment = wandb.init(project='Rico-U-Net', resume='allow', anonymous='must')
experiment.config.update(
    dict(epochs=NUM_EPOCHS, batch_size=BATCH_SIZE * ACCUMULATION_STEPS, learning_rate=LEARNING_RATE,
            training_size = len(train_dataset),
            validation_size = len(val_dataset),
            save_checkpoint=SAVE_CHECKPOINT, amp=AMP)
)
# [optional] finish the wandb run, necessary in notebooks                                                                      
wandb.finish()
```

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
