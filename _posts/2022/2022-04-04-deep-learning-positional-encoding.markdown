---
layout: post
title: Deep Learning - Positional Encoding
date: '2022-04-05 13:19'
subtitle: Transformer Accessory
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## What is Positional Encoding

In [self attention](./2022-03-28-deep-learning-multi-headed-self-attention.markdown), we calculate weights for all embeddings in queries, keys and values. However, word order is also important. E.g., "I ride bike" is not the same as "bike ride I".

Given an input sequence `X0, X1 ... Xn`, we want to find a time encoding such that:

- the time encoding represents the order of time
- the time encoding value is smaller than the embedding space
- each input has a unique encoding
- time encoding dimension should be the same as the input dimension

We arrange the input sequence into an `nxd` vector

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/40f50a34-9803-4f52-a2c3-defea9863d6c" height="200" alt=""/>
       </figure>
    </p>
</div>

For time `n`, embedding_dimension `d` columns `2j` and `2j+1`, the encodings are:

$$
\begin{gather*}
encoding(i, 2j) = sin(\frac{i}{10000^{(2j/d)}})
\\
encoding(i, 2j+1) = cos(\frac{i}{10000^{(2j/d)}})
\end{gather*}
$$

Now let's enjoy some code:

```python
class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_input_timesteps,hidden_size) -> None:
        super().__init__()
        # Adding 1 to make sure this is a batch
        self.time_encodings = torch.zeros((1, max_input_timesteps, hidden_size))
        # i / 10000^(2j)
        coeffs = torch.arange(max_input_timesteps, dtype=torch.float32).reshape(-1, 1) #(max_input_timesteps, 1)
        coeffs = coeffs/torch.pow(
            10000, torch.arange(0, hidden_size, 2, dtype=torch.float32) / hidden_size)  #(max_input_timesteps, 4)
        self.time_encodings [:, :, 0::2] = torch.sin(coeffs)
        self.time_encodings [:, :, 1::2] = torch.cos(coeffs)    #(max_input_timesteps, 4)
        print(f'Rico: {self.time_encodings.shape}')

    def forward(self, X):
        # :X.shape[1] is to because X might be of a different length (lower than max_input_timesteps)
        X = X + self.time_encodings[:, :X.shape[1], :].to(X.device)
        return X

pe = PositionalEncoding(max_input_timesteps=10, hidden_size=4)
X = torch.rand((10, 4))
pe(X)
```

So, we can see that for a given column, embeddings at different timesteps change periodically. Elements Different columns could have the same values as well, but they vary at different frequencies. For the same `i`, the frequency component in sin and cos values decrease.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/4d1f0d07-4721-452c-abb7-dd0229645c1a" height="200" alt=""/>
       </figure>
    </p>
</div>

```python
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
```
