---
layout: post
title: Deep Learning - Bert 
date: '2022-04-10 13:19'
subtitle: 
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Introduction

Bert (BiDirectional Encoder Representation Transformer) is great for tasks like question-answering, NER (Named Entity Recognition), sentence classification, etc.
Bert is not a translation model, because it does not have a decoder that takes in output embedding. Bert started the "Pretraining + fine-tuning" style

## Model Structure

The main differences of BERT from Transformer are:

- Encoder Only
- Training with Segment Embedding
  - Depending on tasks, we might want to add the "segment" information to the input embedding. E.g.,
    - If the task is entailment, we have a premise and a hypothesis. If the premise is true, the hypothesis is true. In that case, we have two segments, and we want to give embeddings to both of them so the logic relationship is clearer in learning

            ```
            Premise: my cat loves staring at the sky. (Segment 1)
            Hypothesis: my cat loves static activities  (Segment 2)
            ```

    - If the task is sentiment analysis, we just need one segment, "My cat is cute"
- The addition of position embedding
  - BERT uses **trainable position embeddings** that are initialized with sinusoidal positional encoding

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/1dc97bc8-3a03-47bb-bd10-8de5aac9c133" height="300" alt=""/>
    </figure>
</p>
</div>

```python
import torch
import math
class PositionalEmbedding(torch.nn.Module):

    def __init__(self, d_model, max_len=128):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        # TODO?
        pe.requires_grad = False
        for pos in range(max_len):   
            # for each dimension of the each position
            for i in range(0, d_model, 2):   
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        # include the batch size
        self.pe = pe.unsqueeze(0)   
        print("self.pe.requires_grad: ", self.pe.requires_grad)
    def forward(self, x):
        return self.pe

class BERTEmbedding(torch.nn.Module):
    def __init__(self, d_model,max_len, vocab_size, dropout_rate):
        super().__init__()
        self.positional_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
        # <PAD> is zero, so it doesn't contribute to gradient
        self.token_embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=0)
        # Assuming we have question-answer.
        self.segment_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=d_model, padding_idx=0)
        self.dropout = torch.nn.Dropout(p = dropout_rate)

    def forward(self, x):
        # x is [<CLS>, <token1>...<SEP>]
        x = self.token_embedding(x) + self.positional_embedding(x) + self.segment_embedding(x)
        return self.dropout(x)
```

### Encoder Only Architecture

- [`Gelu` is used instead of relu](./2022-01-08-deep-learning-Activation-Losses.markdown)

```python
class FeedForward(torch.nn.Module):
    "Implements FFN, a.k.a MLP (multilevel perceptron)."

    def __init__(self, d_model, middle_dim=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out

class BERT(torch.nn.Module):
    def __init__(self, d_model, num_heads, max_len, vocab_size, num_encoder_layers,  dropout_rate):
        super().__init__()
        self.embedding = BERTEmbedding(
            d_model=d_model, 
            max_len=max_len, 
            vocab_size=vocab_size, 
            dropout_rate=dropout_rate)
        self.built_in_encoder_layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model,
                dropout=dropout_rate,
                activation='gelu'
            ) for _ in range(num_encoder_layers)
        ])
        
    def forward(self, x, segment_label, attn_mask, key_padding_mask):
        # segment_info?
        x = self.embedding(x, segment_label)
        for encoder_layers in self.built_in_encoder_layers:
            x = encoder_layers(
                src=x, src_mask=attn_mask, src_key_padding_mask=key_padding_mask
            )
        return x
```

### Masked Language Modelling

When fiding the embeddings for words in the sentence "The animal didn’t cross the street because it was too tired", the noun it can refer to either the animal, or the street. But from the word "tired" we know that it can't be the street. This is a great example of forward-backward context.

- A conventional transformer can "technically" consider forward and backward contexts. However,  while we train them, we need a causal mask so it doesn't look at the later information
- In BERT, because we are looking at the entire input batch, we want to mask out certain parts of the input text and make the model guess the correct tokens

Apply a mask that will mask out 15% of the vocabulary randomly:

- 80% of the time, replace the word with token `[MASK]`. `my dog is hairy → my dog is [MASK]`
- 10% of the time, the word stays unchanged.
- 10% of the time, the word is replaced with another random word. `my dog is hairy → my dog is apple`. We have this strategy so that in fine tuning, we don't need `[MASK]` but can still have decent training.

Code sample: TODO

## Fine Tuning

TODO

## References

- [This ZhiHu Post](https://zhuanlan.zhihu.com/p/103226488)
