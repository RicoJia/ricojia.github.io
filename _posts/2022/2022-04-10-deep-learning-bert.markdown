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

Bert started the "Pretraining + fine-tuning" style

## Model Structure

The main differences of BERT from Transformer are:

- Encoder Only
- The addition of position and segment embedding
    - BERT uses trainable position embeddings that are initialized with sinusoidal positional encoding

Let's enjoy some code

```python
## Position Embedding 
class Encoder(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_vocab_dim,
        encoder_layer_num,
        num_heads,
        max_sentence_length,
        dropout_rate=0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.positional_encoder = OGPositionalEncoder(
            max_sentence_length=max_sentence_length, embedding_size=self.embedding_dim
        )
        self.embedding_converter = torch.nn.Embedding(
            num_embeddings=input_vocab_dim, embedding_dim=self.embedding_dim
        )
        self.dropout_pre_encoder = torch.nn.Dropout(p=dropout_rate)
        self.encoder_layers = torch.nn.ModuleList(
            [
                EncoderLayer(
                    embedding_dim=self.embedding_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                )
                for _ in range(encoder_layer_num)
            ]
        )
```

- Why segment embedding, position embedding

## References

- [This ZhiHu Post](https://zhuanlan.zhihu.com/p/103226488)