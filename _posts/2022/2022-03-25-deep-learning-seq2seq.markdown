---
layout: post
title: Deep Learning - Sequence to Sequence Models
date: '2022-03-25 13:19'
subtitle: seq2seq, encoder-decoder architecture, beam model, Bleu Score
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
        <img src="https://github.com/user-attachments/assets/ab056842-81cf-4e79-a4cc-ddff7b0f79b0" height="200" alt=""/>
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

Sentence normalization is to convert uncommon words into common ones, such as lowercasing, removing punctuation, etc. Without sentence normalization, machine translations tend to output very short sequences. Why?

- The inconsistent vocabulary disrupts the model to learn sequences.

### Seq2Seq

Ilya Sutskever et al in 2014 propsed the (sequence to sequence) seq2seq model. Below is a summary of [the Pytorch page](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html).  Imagine that we are trying to create a machine translator for `French -> English`

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/61e3680c-4c5f-4178-a620-bc7b8d314db0" height="200" alt=""/>
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
        <img src="https://github.com/user-attachments/assets/9ae13184-fd96-445c-9f61-dba8fe972529" height="200" alt=""/>
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
        # self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first = True)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p = 0.1)

    def forward(self, input_batch):
        # [batch_size, max_length, hidden_size]
        embedded = self.dropout(self.embedding(input_batch))
        # run the entire sequence
        # outputs is PackedSequence, hidden and cell: [num_layers, batch_size, hidden_dim]
        # outputs, (hidden, cell) = self.lstm(embedded)
        outputs, hidden = self.gru(embedded)
        # return outputs, (hidden, cell)
        return outputs, hidden
```

#### Decoder

A decoder is another RNN that takes in the context vector and outputs a word sequence to create the translation. At every step, the decoder is given an input token. The initial token is `<SOS>`, and the input token is the output from the last step

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/ecd8d962-bc6b-433e-8f89-ff19cc195d15" height="300" alt=""/>
       </figure>
    </p>
</div>

```python
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.device = torch.device('cuda' if cuda.is_available() else 'cpu')
        self.dropout = nn.Dropout(p = 0.1)


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
        # out = self.dropout(out)
        # out: torch.Size([17, 1, 128])

        # Why do we need a relu here? In many seq2seq implementations, this can be omitted.
        out = F.relu(out)
        # What's the function of the LSTM here? It captures dependencies and generate hidden states that encode information of the sequence.
        # hidden here is (a, C)
        # out, hidden = self.lstm(out, hidden)
        out, hidden = self.gru(out, hidden)
        out = self.out(out)
        # output is: [batch_size, 1, output_dim]. 1 is for each time
        return out, hidden
```

Review Points:

- Teacher enforcing is the technique that feeds the ground truth word as the next timestep input.
- Here we are only feeding the `(hidden state, cell state)`of the encoder into the decoder.
- `torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None)` chooses the k largest elements of the given input, along `dim`. If no `dim` is specified, the last dim is chosen.
- `relu` did not yield any significant change in my experiments.

### Evaluation Outcome

With `LSTM`, the model almost doesn't work. This is after trying all combinations of

- turning on / off `ReLu` in the decoder
- turning on / off `dropout` in the decoder

With `GRU`, the model is a lot better. I saw around 30% exact / close matches. `ReLu` and `dropout` do not create a significant difference in accuracy.

Honestly, it's definitely not good enough to be in production.

Some examples from the `spa-eng` dataset:

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

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/52e0731f-04fb-43de-8ff3-16eb04b84085" height="200" alt=""/>
       </figure>
    </p>
</div>

In this case, the encoder is a pretrained AlexNet, where the last softmax layer is removed, so we can get a 4096 feature vector. Then, this feature vector is fed into one (TODO: many?) RNN cells and finally output a sentence.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/78e8f6ee-5849-4b75-9224-9f13a5a99516" height="200" alt=""/>
       </figure>
    </p>
</div>

## Picking The Most Likely Sentence

Since the decoder directly outputs the probablity of `p(Output Lagunage Sentence| Input Language Sentence)`, we want to choose the sentence that maximizes this probability.

Naively, one can choose the word that maximizes its current probability. That is, pick the best first word, pick the second best word, etc. However, the greedy approach doesn't alwasy work. E.g., between "Jane is visiting Africa" vs "Jane is going to Africa", the first sentence is less verbose and better. However, "Jane is going" has a higher local probability than "Jane is visiting" since "going" is a more common word.

So one way is to search in the sentence space with the same length for the sentence that maximizes this probability. That however, needs some approximation.

### Beam Search

One approach is "beam search". The idea is, at each step, we are given the K probable candidates sequences. Each candidate will be fed into the model, and get k probable current words (with probability being the raw output). Then, we get the total probability of all the new sequences, and choose the top K sequences. E.g.,

1. At time 1, the encoder gives us an embedding `e`. We feed `e` into the decoder, and get a raw probability across all words, `y1`

- Based on `y1`, we choose 3 most likely candidate: `["Jane", "In", and "September"]`.

2. At time 2, we feed `["Jane", "In", and "September"]` into the model.

- We have the possible sequences

    ```
    [
        "Jane is", "Jane goes, "Jane does",
        "In September", "In space", "In car",
        "September is", "September goes, "September comes"
    ] 
    ```

- In the mean time, we can calculate each new word `i`'s probability `y2_i`. The total probability of the sequence of `i` is `y2_i * y1`. This is equivalent to $p(y2_i\| y1) p(y1) = p(y2_i, y1)$
- We decided that the top **3** most probable sequence is:

    ```
    [
        "Jane is", "Jane goes, "In September", 
    ] 
    ```

3. At time 3, we feed `["Jane is", "Jane goes, "In September"]` into the model and get 9 most probable sequences again. We rate their total probability `p(y3_j, y2_i, y1)` and keep the top 3 sequences.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/297c5656-0861-40fe-aceb-50cbdbf19468" height="300" alt=""/>
            <figcaption><a href="https://commons.wikimedia.org/wiki/File:Beam_search.gif">Source: Wikimedia Commons </a></figcaption>
       </figure>
    </p>
</div>

Side Notes:

- The name "beam search" comes from the analogy to "illuminating the top k nodes". From the above illustration, one can see that the search is actually a tree
- One trick is to add up the log probability at time `t` to avoid underflow. This value is always negative.

$$
\begin{gather*}
\sum_{T} log(y^{(t)}) = \sum_{T} log(P(y^{(t)} \| y^{(t-1)} ... y^{(0)}))
\end{gather*}
$$

- This model unnaturally has a tendency to prefer shorter sentences, because the more words, the lower total probability. So in practice, people use:

$$
\begin{gather*}
\frac{1}{m}\sum_{T} log(y^{(t)})
\end{gather*}
$$

- Beam width B is usually 10 for production. 100 already is quite large. 1000 is mostly for reseach.

## Error Analysis Of Beam Search vs RNN

When we have trained a model and look at the trained output of some sample sequences, we will compare the output of the model `y_pred` against the human-provided translation: `y*`.
It's always tempting to collect more training data, try a different model, or go back to the RNN implementation and see if anything was implemented wrong. But it's also possible that our Beam-search implementation might not produce the best result. In the later case, we might want to try different Beam Widths, or even try a different sampling approach.

A quick way to determine whether the beam search is to evaluate if `prob(y_pred) < prob(y*)`. For example, if prediction is `"Jane is going to Africa"`, and the human groudtruth is `Jane went to Africa`. So,

1. Feed "Jane" into the decoder network get `y_1p`, `y_1*`
2. Feed "is" and "went" into the decoder `y_2p`, `y_2*`
3. Feed "goint" and "went" into the decoder `y_3p`, `y_3*`
...
4. Calculate for timestep `T`,  $P_p = \Pi_T y_p^{(t)}$, $P_* = \Pi_T y_*^{(t)}$.
    - If $P_p > P_*$, then the RNN is at fault
    - If $P_p < P_*$ then increasing beam width might help

## Bleu Score

In 2002, Papineni et al. proposed "Bleu score", a single number that decently well characterizes the performance of systems with multiple references groundtruth like image captioning. `Bleu` is a **precision**

E.g., if we have two references,

```
- Reference 1: The cat is on the table
- Reference 2: There is a cat on the table
- MT Output: The cat the cat is on the table
```

We can look at how many single words (uni-gram) or word pairs (bi-gram) in the MT output sentence actually appeared in the sentence.

For bi-gram, we count the number of each word pair's appearances in the MT output (in the `Count` column), and the total number of appearances in the two references combined (in the `Count<sub>clip</sub>` column)

| Phrase   | Count | Count<sub>clip</sub> |
|----------|-------|----------------------|
| the cat  | 2     | 1                    |
| cat the  | 1     | 1                    |
| cat on   | 1     | 0                    |
| on the   | 1     | 1                    |
| the mat  | 1     | 1                    |

`Bleu2 = count_clip / count = 4/6`

Similarly, we count the number of single words that appear in the MT output and in the two references combined.

`Bleu1 = count_clip_1_gram / count_1_gram`

We can combine the result of the `Bleu` score on multiple n_grams.

$$
\begin{gather*}
score = BP \cdot exp(0.25 \sum_N P_n)
\\
BP = 1 \text{if MT_output_length > reference_output_length}
\\ or
\\
BP = exp(1- reference_output_length/MT_output_length )
\end{gather*}
$$

YOU CAN FIND SOME GOOD OPEN SOURCE IMPLEMENTATIONS ON THIS!

## References

[1] Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. Advances in Neural Information Processing Systems (NIPS), pp. 3104-3112.

[2] Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1724-1734.

[3] Mao, J., Xu, W., Yang, Y., Wang, J., Huang, Z., & Yuille, A. (2014). Deep captioning with multimodal recurrent neural networks. Proceedings of the International Conference on Learning Representations (ICLR).

[4] Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2014). Show and tell: Neural image caption generator. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3156-3164.

[5] Karpathy, A., & Fei-Fei, L. (2015). Deep visual-semantic alignments for generating image descriptions. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3128-3137.

[6] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. BLEU: A Method for Automatic Evaluation of Machine Translation. In Proceedings of the 40th Annual Meeting on Association for Computational Linguistics (ACL ’02), pages 311–318. Association for Computational Linguistics. <https://doi.org/10.3115/1073083.1073135>
