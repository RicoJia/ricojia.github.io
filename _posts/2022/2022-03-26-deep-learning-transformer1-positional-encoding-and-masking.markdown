---
layout: post
title: Deep Learning - Transformer Series 1 - Embedding Pre-Processing
date: '2022-03-26 13:19'
subtitle: Positional Encoding, Padding Mask, Look-ahead Mask
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## What is Positional Encoding

In [self attention](./2022-03-28-deep-learning-multi-headed-self-attention.markdown), we calculate weights for all embeddings in queries, keys and values. However, word order is also important. E.g., "I ride bike" is not the same as "bike ride I".

Given an input sequence `X0, X1 ... Xn`, we want to find a time encoding such that:

- the time encoding represents the order of time
- the time encoding value is **smaller than the embedding space**. Otherwise, the encoding could distort the semantic embeddings. `sine` and `cosine` are great since they are only within `[-1, 1]`.
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

For time `i`, embedding_dimension `d` columns `2j` and `2j+1`, the encodings are:

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

In below's chart, 50 128-dimension positional encodings are shown. Each row is the index of the encoding, each column is a number in a 128-dimension vector.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/9e5826c7-fa69-4cbe-8934-4716e4fc6dae" height="300" alt=""/>
    </figure>
</p>
</div>

For example, for the `50th` input embedding, the 0th dim corresponds to the value `sin(50/10000^{(2*0/128)}})`. The 127th dim corresponds to `cos(50/10000^(126/128))`. As we can see, the frequency of encoding "bit" changing decreases, as the dimension number goes higher.

## Masking

There are two types of masking for building a transformer: **padding mask** and **look-ahead mask**

### Padding Mask

Sometimes, the input exceeds the maximum sentence length of our network. For example, we might have input

```
[["Do", "you", "know", "when", "Jane", "is", "going", "to", "visit", "Africa"], 
 ["Jane", "visits", "Africa", "in", "September" ],
 ["Exciting", "!"]
]
```

Which might get vectorized as:

```
[[ 71, 121, 4, 56, 99, 2344, 345, 1284, 15],
    [ 56, 1285, 15, 181, 545],
    [ 87, 600]
]
```

In that case, we want to:

- Truncate the sequence to uniform length
- Pad a large negative number (-1e9) instead of 0 onto short sequences. Why -1e9? Because later in scaled-dot product attention, if we have large negative values, $softmax(\frac{QK}{\sqrt(d_k)} V)$ will likely give probabilities of zero

```
[[ 71, 121, 4, 56, 99], 
 [2344, 345, 1284, 15, -1e9],
 [ 56, 1285, 15, 181, 545],
 [ 87, 600, -1e9, -1e9, -1e9]
]
```

To illustrate:

```python
def create_padding_mask(padded_token_ids):
    # We assume this has been trucated, then padded with 0 (for short sentences)
    # [Batch_size, Time]
     mask = (padded_token_ids != 0).float() 
     return mask
# Sample input sequences with padding (batch_size, seq_len)
input_seq = torch.tensor([
    [5, 7, 9, 0, 0],    # Sequence 1 (padded)
    [3, 2, 4, 1, 0],    # Sequence 2 (padded)
    [6, 1, 8, 4, 2]     # Sequence 3 (no padding)
])
padding_mask = create_padding_mask(input_seq)
# see the zeros in input_seq will also become 0 in softmax
print(torch.nn.functional.softmax(input_seq + (1 - padding_mask) * -1e9))
```

[The multi-headed attention implemented in Keras](https://keras.io/api/layers/attention_layers/multi_head_attention/) was implemented this way.

### Look-ahead Mask

Given a full sequence, we want to prevent the model from "cheating" by looking at future tokens during training. In autoregressive models, like language models, when predicting a word, the mdoel should only consider the current and previous tokens, not future ones. 

```python
def create_look_ahead_mask(sequence_length):
    """
    Return an upper triangle
    tensor([[False,  True,  True],
            [False, False,  True],
            [False, False, False]])
    """
    # diagonal = 0 is to include the diagonal items
    return (1- torch.tril(torch.ones(sequence_length, sequence_length), diagonal=0)).bool()
```