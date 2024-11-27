---
layout: post
title: Deep Learning - Transformers
date: '2022-04-05 13:19'
subtitle: TODO
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Overview

[We've seen that RNN and CNN has a longer maximum path length](./2022-03-28-deep-learning-multi-headed-self-attention.markdown). CNN could have better computational complexity for long sequences, but overall, self attention is the best for deep architectures. The transformer depends solely on self attention, and does not have convolutional or recurrent layers, unlike its predecessors, like `seq2seq` [1].

Transformer was proposed for sequence-to-sequence learning on text data, but it's gained popularity in speech, vision, and reinforcement learning tasks as well.

The Transformer has an encoder-decoder architecture.

- Different from Bahdanau Attention, input is added with positional encoding before being fed into the encoder and the decoder

### Encoder

The encoder has one multi-head self-attention pooling and one positionwise feed-forward network (FFN) modules. Some highlights are:

- In the multi-head self-attention pooling, the queries, keys, and values are the previous encoder output.
- Inspired by ResNet, **a residual connection (or skip connection) is added to boost the input signal**. Because of this, **no layer in the encoder changes the shape of the input**.
- The positionwise feedforward networks **transforms embeddings at all timesteps using the same multi-layer perceptrons (MLP)**. So, they do not perform any time-wise operations.
  - Positionwise FFN **COULD** have a different hidden layer dimension within itself, as shown below. It just needs to output the same dimension.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/195ef3b9-35fe-4c4d-a582-bcc9088c0f92" height="500" alt=""/>
    </figure>
</p>
</div>

Now, let's enjoy some code.

#### Positionwise FFN

```python
import torch

class PositionwiseFFN(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim) -> None:
        super().__init__()
        self.dense1 = torch.nn.LazyLinear(hidden_dim)
        self.relu = torch.nn.ReLU()
        self.dense2 = torch.nn.LazyLinear(output_dim)
    def forward(self, X):
        # (batch size, number of time steps, output_dim).
        return self.dense2(self.relu(self.dense1(X)))

ffn = PositionwiseFFN(4, 8)
ffn.eval()
# see (2, 3, 8)
print(ffn(torch.ones((2, 3, 4))).shape)
```

- previous encoder output? TODO

### Decoder

The decoder also has residual connections, normalizations, two attention pooling modules, and one positionwise FFN module. Some highlights are:

- The first attention module is a self-attention module.
  - Its queries, keys and values are all from the decoder.
  - This masked attention preserves the autoregressive property, ensuring that the prediction only depends on those output tokens that have been generated. TODO: what does this mean? What is autoagressive?
- The attention module between the first self-attention module and the positionwise FFN module is called **"encoder-decoder attention".**
  - Queries are from the decoder's self-attention layer
  - Keys and values are from the encoder.

- decoder output -> linear layer -> softmax layer to predict the next word one word at a time

## TODO

![Screenshot from 2024-11-17 15-52-10](https://github.com/user-attachments/assets/c293b115-a8f8-42a0-9589-fe6a1b200beb)

- A Transformer Network processes sentences from left to right, one word at a time.

- What does the output of the encoder block contain?
- Why is positional encoding important in the translation process? (Check all that apply)
- Which of these is not a good criterion for a good positional encoding algorithm?
  - must be deterministic
  - can generalize to longer sentences
  - A common encoding for each timestep (words position in a sentence)

##

## References

[1] [Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems (pp. 5998–6008).](https://arxiv.org/pdf/1706.03762)
