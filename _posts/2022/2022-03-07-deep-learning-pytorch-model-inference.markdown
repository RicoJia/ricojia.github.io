---
layout: post
title: Deep Learning - PyTorch Model For Inferencing
date: 2022-03-07 13:19
subtitle: eval mode
comments: true
header-img: img/home-bg-art.jpg
tags:
  - Deep Learning
---
---

## Eval Mode

For model training, it's common to see below:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model = Encoder(...).to(device).eval()  # to(device) is always a safer alternative than .cuda() because GPU might not be available
  
with torch.no_grad():  
    out = model(x.to(device), feats.to(device))
```

- `eval()` would disable Batch Norm and `Dropout` calculation. This is **very important** for inferencing. In the meantime, `eval()` does not turn off gradient tracking. So we need `with torch.no_grad()` for this purpose.  
 	- Recall that `Dropout` randomly zeros out activations
