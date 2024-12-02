---
layout: post
title: Deep Learning - Transformer All Together
date: '2022-04-05 13:19'
subtitle: Encoder, Decoder
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

[A good custom implementation is here](https://github.com/datawhalechina/learn-nlp-with-transformers/blob/main/docs/%E7%AF%87%E7%AB%A02-Transformer%E7%9B%B8%E5%85%B3%E5%8E%9F%E7%90%86/2.2-%E5%9B%BE%E8%A7%A3transformer.md)

### Encoder

The encoder has one multi-head self-attention pooling and one positionwise feed-forward network (FFN) modules. Some highlights are:

- In the multi-head self-attention pooling, the queries, keys, and values are the previous encoder output.
- Inspired by ResNet, **a residual connection (or skip connection) is added to boost the input signal**. Because of this, **no layer in the encoder changes the shape of the input**.
- The positionwise feedforward networks **transforms embeddings at all timesteps using the same multi-layer perceptrons (MLP)**. So, they do not perform any time-wise operations.
  - Positionwise FFN **COULD** have a different hidden layer dimension within itself, as shown below. It just needs to output the same dimension.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/7430abaf-428e-4e58-acde-77ce5e8b65ab" height="500" alt=""/>
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

### The full encoder output

- Scaling: the embeddings are scaled by $\sqrt{\text{embedding\_dimension}}$" before adding positional encodings so their magnitudes match. There's a [StackExchange thread on why exactly this is needed](https://datascience.stackexchange.com/questions/87906/transformer-model-why-are-word-embeddings-scaled-before-adding-positional-encod). However, some were also wondering about its necessity

### Decoder

The decoder also has residual connections, normalizations, two attention pooling modules, and one positionwise FFN module. Some highlights are:

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/4f1b271f-8e3a-432d-97a2-6ce6ec89db35" height="500" alt=""/>
       </figure>
    </p>
</div>

- The first attention module is a self-attention module.
  - Its queries, keys and values are all from the decoder.
  - It uses an **lookahead mask, or attention mask**, which preserves the autoregressive property, ensuring that the prediction only depends on those output tokens that have been generated.
- The attention module between the first self-attention module and the positionwise FFN module is called **"encoder-decoder attention".**
  - This layer uses a padding mask.
  - Queries are from the decoder's self-attention layer
  - Keys and values are from the encoder.

## All Together

```python
def forward(self, input_sentence, output_sentence, enc_padding_mask, attn_mask, dec_padding_mask):
    # input_sentence: [Batch_Size, input_sentence_length]
    # [batch_size, input_seq_len, qk_dim]
    enc_output = self.encoder(X=input_sentence, enc_padding_mask=enc_padding_mask) 
    # [batch_size, output_seq_len, qk_dim]
    dec_output = self.decoder(X = output_sentence, enc_output=enc_output,
                                attn_mask = attn_mask, key_padding_mask = dec_padding_mask)
    # This is basically the raw logits
    # THIS IS ASSUMING THAT WE ARE USING CROSS_ENTROPY LOSS
    # [batch_size, output_seq_len,target_vocab_dim]
    logits = self.final_dense_layer(dec_output)
    return logits
```

- At the end, we want the probabilities across target language words, so softmax is needed for training. However, ReLu is not advised here, because it could distort the relative differences between logits by setting the negative ones to 0. The standard practice is: **No ReLu between Linear and Softmax.**
- Also, THIS IS ASSUMING THAT WE ARE USING CROSS_ENTROPY LOSS. So here we are not adding a softmax layer here.

![Screenshot from 2024-11-17 15-52-10](https://github.com/user-attachments/assets/c293b115-a8f8-42a0-9589-fe6a1b200beb)

### Advantages and Disadvantages of Transformer

Advantages:

- Parallel computing. Transformer abandoned the CNN and RNN architectures that were used for decades.
  - The input is `[batch_size, input_seq_len, input_vocab_dim]`, the output is `[batch_size, output_seq_len,target_vocab_dim]`. So unlike RNN architecutres which parse a sequence step by step, attention pooling with multiple heads (or partitions of attention) in parallel.

Disadvantages:

- Local feature extraction (like in CNN) is lacking.
-

## Tasks and Data

It's common practice to pad input sequences to `MAX_SENTENCE_LENGTH`. Therefore,

- the input is always [batch_size, max_sentence_length]
- `NUM_KEYS = NUM_QUERIES = max_sentence_length` since neither the encoder nor the decoder changes the `max_sentence_length` dimension

In practice, one can apply below methods to reduce padding:

- Bucketing - bucketing is to group sentences of similar lengths to reduce sentence lengths.
- Packed Sequences: PyTorch's `pack_padded_sequence` and `pad_packed_sequence` utilities (more common in RNNs) to handle variable-length sequences.

Applications

- Machine Translation (using World-Machine-Translation datasets)
- Named Entity Recognition (like extracting "phone number" from resumes)

## References

[1] [Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems (pp. 5998–6008).](https://arxiv.org/pdf/1706.03762)
