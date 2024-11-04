---
layout: post
title: Deep Learning - Sequence to Sequence Models
date: '2022-03-25 13:19'
subtitle: seq2seq
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
    - Hands-on
---

## Sequence to Sequence Models: The Encoder - Decoder Architecture

### Machine Translation

Early sequence models use two RNN/LSTM cells to create an encoder-decoder architecture for machie translation. This architecture works decently well. E.g., a French sentence comes in, an English sentence comes out.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/ab056842-81cf-4e79-a4cc-ddff7b0f79b0" height="300" alt=""/>
    </figure>
</p>
</div>

Advantages:

- This method works well when the input sequence is not too long.

Key differences between machine translation and languange models:

- **Machine Translation needs the most likely output, instead of randomly chosen based on a probability distribution.**
- The encoder network does not always start from zeros. Instead, it figures out the **whole sentence's embedding**
- The decoder directly outputs the probablity of `p(Output Lagunage Sentence| Input Language Sentence)`

Multiple groups came up with this architecture and alike in 2014 and 2015 [3], [4], [5].

### Seq2Seq
Ilya Sutskever et al in 2014 propsed the (sequence to sequence) seq2seq model. Below is a summary of [the Pytorch page](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html).  Imagine that we are trying to create a machine translator for `French -> English`

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/61e3680c-4c5f-4178-a620-bc7b8d314db0" height="300" alt=""/>
        <figcaption><a href="">Source: </a></figcaption>
    </figure>
</p>
</div>

#### Encoder

The encoder:

1. First learns the embedding of French word
2. Then outputs **the context vector**, or the embedding of the whole sentence in the `N-dim` space. 

Note that I'm using LSTM but a GRU can also be used. For every input, the RNN outputs a vector, and a hidden state. We pass down the outputs, the hidden state, and the cell state to the decoder. 


<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/9ae13184-fd96-445c-9f61-dba8fe972529" height="300" alt=""/>
    </figure>
</p>
</div>

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers=1):
        super(Encoder, self).__init__()
        # This is the embedding for each word?
        self.embedding = nn.Embedding(input_dim, embed_dim)
        # The output an embedding of the sentence
        hidden_dim = embed_dim
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first = True)
        self.dropout = nn.Dropout(p = 0.1)

    def forward(self, input_batch):
        # [batch_size, max_length, hidden_size]
        embedded = self.dropout(self.embedding(input_batch))
        # run the entire sequence
        # hidden and cell: [num_layers, batch_size, hidden_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)
```

#### Decoder

A decoder is another RNN that takes in the context vector and outputs a word sequence to create the translation. At every step, the decoder is given an input token. The initial token is `<SOS>`, and the input token is the output from the last step

![decoder-network](https://github.com/user-attachments/assets/ecd8d962-bc6b-433e-8f89-ff19cc195d15)

```python
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.device = torch.device('cuda' if cuda.is_available() else 'cpu')

    def forward(self, encoder_outputs, encoder_hidden, target_tensor = None):
        batch_size = encoder_outputs.shape[0]
        # How to define SOS token?
        # Creates [batch_size, 1] uninitialized tensor sequence.
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        for i in range(MAX_LENGTH):

            decoder_output, decoder_hidden = self.get_word_embedding(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            # Teacher forcing: We feed the target as the next input. 
            if target_tensor is not None:
                decoder_input = target_tensor[:,i].unsqueeze(1)
            # Without teacher forcing, we feed the current output as the next input
            else: 
                # Here without a target token, get the most 1 probable token and its index (topi) using topk(1). 

                _, topi = decoder_output.topk(1)
                # why do we want to detach? Does it create an alias variable that accesses the same memory?
                # Detach is to detach the tensor from the computational graph, so in backprop, the gradient here won't be calculated.
                decoder_input = topi.squeeze(-1).detach()
        # why concat along dim 1? Because the output is [batch_size, 1, output_dim]. 1 is for each time. So concat will result in [batch_size, MAX_LENGTH, output_dim]
        decoder_outputs = torch.cat(decoder_outputs, dim = 1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def get_word_embedding(self, input, hidden):
        # So we are outputting a token for every output length, right? Yes.
        # Then, what does going through the embedding mean? It means learning the embedding of the destination language. 
        # encoder hidden: torch.Size([1, 17, 128]), cell: torch.Size([1, 17, 128])
        out = self.embedding(input)
        # out: torch.Size([17, 1, 128])

        # Why do we need a relu here? In many seq2seq implementations, this can be omitted.
        # out = F.relu(out)
        # What's the function of the LSTM here? It captures dependencies and generate hidden states that encode information of the sequence.
        # hidden here is (a, C)
        out, hidden = self.lstm(out, hidden)
        out = self.out(out)
        # output is: [batch_size, 1, output_dim]. 1 is for each time
        return out, hidden
```

Review Points:

- Teacher enforcing is the technique that feeds the ground truth word as the next timestep input.
- Here we are only feeding the `(hidden state, cell state) `of the encoder into the decoder.
- `torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None)` chooses the k largest elements of the given input, along `dim`. If no `dim` is specified, the last dim is chosen.
- `relu` did not yield any significant change in my experiments.

### Evaluation Outcome

Honestly, it's definitely not good enough to be in production. Some examples from the `spa-eng` dataset:

```
> estoy muy emocionada
= i m very excited
< i am a boy <EOS>

> eres un imbecil
= you re an imbecile
< you re my imbecile <EOS>

> estoy algo impactado
= i m kind of stunned
< a donut <EOS>
```

The exact matches are quite rare. I haven't count, but most sentences might have some keywords but do not make full sense. The final training loss is fairly small - it was only `0.001`. I'm suspecting this is due to overfitting. 

### Image Captioning

This can be similarly used in **image captioning**, where the input is an RGB image, the output is a **short sentence describing the image,** like "A cat sits in a chair". 

![Screenshot from 2024-11-03 10-22-18](https://github.com/user-attachments/assets/52e0731f-04fb-43de-8ff3-16eb04b84085)

In this case, the encoder is a pretrained AlexNet, where the last softmax layer is removed, so we can get a 4096 feature vector. Then, this feature vector is fed into one (TODO: many?) RNN cells and finally output a sentence.

![Screenshot from 2024-11-03 10-47-37](https://github.com/user-attachments/assets/78e8f6ee-5849-4b75-9224-9f13a5a99516)


## References

[1] Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. Advances in Neural Information Processing Systems (NIPS), pp. 3104-3112.

[2] Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1724-1734.

[3] Mao, J., Xu, W., Yang, Y., Wang, J., Huang, Z., & Yuille, A. (2014). Deep captioning with multimodal recurrent neural networks. Proceedings of the International Conference on Learning Representations (ICLR).

[4] Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2014). Show and tell: Neural image caption generator. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3156-3164.

[5] Karpathy, A., & Fei-Fei, L. (2015). Deep visual-semantic alignments for generating image descriptions. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3128-3137.