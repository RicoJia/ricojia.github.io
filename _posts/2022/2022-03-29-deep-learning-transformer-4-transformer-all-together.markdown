---
layout: post
title: Deep Learning - Transformer Series 4 - Transformer All Together
date: '2022-03-29 13:19'
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

#### Encoder Layer

The encoder layer has 1 multi-head attention (self attention). There are two `Add&Norm` layers. Either takes in a skip connection.
In this layer, we stick to the `embedding_dim`.

```python
class EncoderLayer(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout_rate=0.1,
    ) -> None:
        super().__init__()
        # need dropout. The torch implementation already has it
        self.mha = MultiHeadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
        )
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.ffn = PositionwiseFFN(hidden_dim=embedding_dim, output_dim=embedding_dim)
        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, X, attn_mask, key_padding_mask):
        # Self attention (input_seq_len, batch_size, embedding_dim)
        self_attn_output, self_attn_weight = self.mha(
            X, X, X, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        # apply dropout layer to the self-attention output (~1 line)
        self_attn_output = self.dropout1(
            self_attn_output,
        )
        # Applying Skip Connection
        mult_attn_out = self.layernorm1(
            X + self_attn_output
        )  # (input_seq_len, batch_size, embedding_dim)
        ffn_output = self.ffn(
            mult_attn_out
        )  # (input_seq_len, batch_size, embedding_dim)
        ffn_output = self.dropout2(ffn_output)
        # Applying Skip Connection
        encoder_layer_out = self.layernorm2(
            ffn_output + mult_attn_out
        )  # (input_seq_len, batch_size, embedding_dim)
        return encoder_layer_out
```

### The full encoder output

```python
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

    def forward(self, X, enc_padding_mask):
        # X: [Batch_Size, Sentence_length]
        X = self.embedding_converter(
            X
        )  # X: [Batch_Size, Sentence_length, embedding_size]
        X *= math.sqrt(float(self.embedding_dim))
        # [Batch_Size, Sentence_length, embedding_dim]
        X = self.positional_encoder(X)  # applies positional encoding in addition
        X = self.dropout_pre_encoder(X)

        X = X.permute(1, 0, 2)  # [input_seq_len, batch_size, qk_dim]
        for encoder_layer in self.encoder_layers:
            X = encoder_layer(X, attn_mask=None, key_padding_mask=enc_padding_mask)
        X = X.permute(1, 0, 2)  # [batch_size, input_seq_len, qk_dim]
        return X
```

- Scaling: the embeddings are scaled by $\sqrt{\text{embedding\_dimension}}$" before adding positional encodings so their magnitudes match. There's a [StackExchange thread on why exactly this is needed](https://datascience.stackexchange.com/questions/87906/transformer-model-why-are-word-embeddings-scaled-before-adding-positional-encod). However, some were also wondering about its necessity

### Decoder Layer

A decoder layer has 1 multi-head self attention, and 1 encoder-decoder attention. In this layer, we do not change the embedding dimension, either.

```python
class DecoderLayer(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout_rate=0.1,
    ) -> None:
        super().__init__()
        # need dropout. The torch implementation already has it
        self.mha1 = MultiHeadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
        )
        self.mha2 = MultiHeadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
        )
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)
        self.dropout3 = torch.nn.Dropout(p=dropout_rate)
        self.ffn = PositionwiseFFN(hidden_dim=embedding_dim, output_dim=embedding_dim)
        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self.layernorm3 = torch.nn.LayerNorm(normalized_shape=embedding_dim)

    def forward(self, X, enc_output, attn_mask, key_padding_mask):
        """
        Args:
            X : embedding from output sequence [output_seq_len, batch_size, qk_dim]
            enc_output : embedding from encoder
            attn_mask : Boolean mask for the target_input to ensure autoregression
            key_padding_mask : Boolean mask for the second multihead attention layer

        Returns:
            decoder output:
        """
        self_attn_output, decoder_self_attn_weight = self.mha1(
            X, X, X, attn_mask=attn_mask, key_padding_mask=None
        )
        # apply dropout layer to the self-attention output (~1 line)
        self_attn_output = self.dropout1(
            self_attn_output,
        )
        # Applying Skip Connection
        out1 = self.layernorm1(
            X + self_attn_output
        )  # (output_seq_len, batch_size, embedding_dim)

        self_attn_output, decoder_encoder_attn_weight = self.mha2(
            out1,
            enc_output,
            enc_output,
            attn_mask=None,
            key_padding_mask=key_padding_mask,
        )
        # apply dropout layer to the self-attention output (~1 line)
        self_attn_output = self.dropout2(
            self_attn_output,
        )
        # Applying Skip Connection
        out2 = self.layernorm2(
            out1 + self_attn_output
        )  # (output_seq_len, batch_size, embedding_dim)

        ffn_output = self.ffn(out2)  # (output_seq_len, batch_size, embedding_dim)
        ffn_output = self.dropout3(ffn_output)
        # Applying Skip Connection
        out3 = self.layernorm2(
            ffn_output + out2
        )  # (output_seq_len, batch_size, embedding_dim)
        return out3, decoder_self_attn_weight, decoder_encoder_attn_weight
```

### Decoder

The decoder also has residual connections, normalizations, two attention pooling modules, and one positionwise FFN module.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/4f1b271f-8e3a-432d-97a2-6ce6ec89db35" height="500" alt=""/>
       </figure>
    </p>
</div>

```python
class Decoder(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        target_vocab_dim,
        decoder_layer_num,
        max_sentence_length,
        dropout_rate=0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.positional_encoder = OGPositionalEncoder(
            max_sentence_length=max_sentence_length, embedding_size=self.embedding_dim
        )
        self.embedding_converter = torch.nn.Embedding(
            num_embeddings=target_vocab_dim, embedding_dim=self.embedding_dim
        )
        self.dropout_pre_decoder = torch.nn.Dropout(p=dropout_rate)
        self.dec_layers = torch.nn.ModuleList(
            [
                DecoderLayer(
                    embedding_dim=self.embedding_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                )
                for _ in range(decoder_layer_num)
            ]
        )

    def forward(self, X, enc_output, lookahead_mask, key_padding_mask):
        """
        Args:
            X : [batch_size, output_sentences_length]
            enc_output : [batch_size, input_seq_len, qk_dim].
                TODO: This might be a small discrepancy from the torch implementation, which is [input_seq_len, batch_size, qk_dim]
            lookahead_mask : [num_queries, num_keys]
            key_padding_mask : [batch_size, num_keys]
        """
        #  [batch_size, output_sentences_length]
        X = self.embedding_converter(X)
        X *= math.sqrt(float(self.embedding_dim))
        X = self.positional_encoder(X)  # applies positional encoding in addition
        X = self.dropout_pre_decoder(X)
        X = X.permute(1, 0, 2)  # [output_seq_len, batch_size, qk_dim]
        enc_output = enc_output.permute(1, 0, 2)
        # [num_keys, batch_size, qk_dim]
        decoder_self_attns, decoder_encoder_attns = [], []
        for decoder_layer in self.dec_layers:
            X, decoder_self_attn, decoder_encoder_attn = decoder_layer(
                X,
                enc_output,
                attn_mask=lookahead_mask,
                key_padding_mask=key_padding_mask,
            )
            decoder_self_attns.append(decoder_self_attn)
            decoder_encoder_attns.append(decoder_encoder_attn)
        X = X.permute(1, 0, 2)  # [batch_size, output_seq_len, qk_dim]
        return X, decoder_self_attns, decoder_encoder_attns
```

- The first attention module is a self-attention module.
  - Its queries, keys and values are all from the decoder.
  - It uses an **lookahead mask, or attention mask**, which preserves the autoregressive property, ensuring that the prediction only depends on those output tokens that have been generated.
- The attention module between the first self-attention module and the positionwise FFN module is called **"encoder-decoder attention".**
  - This layer uses a padding mask.
  - Queries are from the decoder's self-attention layer
  - Keys and values are from the encoder.

## All Together

Phew, what a journey! Good job in making it this far. Let's now put all these pieces together. **[All code snippets in this post have been tested against their PyTorch counterparts](https://github.com/RicoJia/Machine_Learning/blob/d4008bb35fc1da89bf3b2273314aff712ad65dab/RicoModels_pkg/ricomodels/tests/test_og_transformer.py)**

```python
class Transformer(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        input_vocab_dim,
        target_vocab_dim,
        layer_num,
        num_heads,
        max_sentence_length,
        dropout_rate=0.1,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            input_vocab_dim=input_vocab_dim,
            encoder_layer_num=layer_num,
            num_heads=num_heads,
            max_sentence_length=max_sentence_length,
            dropout_rate=dropout_rate,
        )

        self.decoder = Decoder(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            target_vocab_dim=target_vocab_dim,
            decoder_layer_num=layer_num,
            max_sentence_length=max_sentence_length,
            dropout_rate=dropout_rate,
        )

        self.final_dense_layer = torch.nn.Linear(
            in_features=embedding_dim,
            out_features=target_vocab_dim,
            bias=False,
        )
        self.final_relu = torch.nn.ReLU()
        self.final_softmax = torch.nn.Softmax(dim=-1)

    def forward(
        self,
        input_sentences,
        output_sentences,
        enc_padding_mask,
        attn_mask,
        dec_padding_mask,
    ):
        # input_sentences: [Batch_Size, input_sentences_length]
        # [batch_size, input_seq_len, qk_dim]
        enc_output = self.encoder(X=input_sentences, enc_padding_mask=enc_padding_mask)
        # [batch_size, output_seq_len, qk_dim]
        dec_output, decoder_self_attns, decoder_encoder_attns = self.decoder(
            X=output_sentences,
            enc_output=enc_output,
            lookahead_mask=attn_mask,
            key_padding_mask=dec_padding_mask,
        )
        # This is basically the raw logits.
        # THIS IS ASSUMING THAT WE ARE USING CROSS_ENTROPY LOSS
        # [batch_size, output_seq_len,target_vocab_dim]
        logits = self.final_dense_layer(dec_output)
        return logits, decoder_self_attns, decoder_encoder_attns
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

## References

[1] [Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems (pp. 5998–6008).](https://arxiv.org/pdf/1706.03762)
[2] [Mislav Jurić Blogpost](https://www.mislavjuric.com/transformer-from-scratch-in-pytorch/)
