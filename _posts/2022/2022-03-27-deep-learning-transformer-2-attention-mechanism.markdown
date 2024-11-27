---
layout: post
title: Deep Learning - Attention Mechanism
date: '2022-03-27 13:19'
subtitle: Bahdanau Attention, Query-Key-Value
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Attention Intuition

Imagine we are sitting in a room. We have a red cup of coffee, and a notebook in front of us. When we first sit down, the red cup stands out. So it attracts our attention "involuntarily" to notice the red cup first.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/ea5a3e03-da76-4c8f-9409-f3cfcde01bbf" height="200" alt=""/>
        <figcaption><a href="https://zh.d2l.ai/chapter_attention-mechanisms/attention-cues.html#id4"> Our eyes are attracted to the red cup. That's involuntary attention </a></figcaption>
    </figure>
</p>
</div>

After drinking the coffee, we tell ourselves that "we need to focus on the notebook now". So we voluntarily and consciously pull our attention to the notebook. Because we are consciously doing it, the attention strength is stronger.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/efd12447-d0f6-466a-844c-7b976d1d90d1" height="200" alt=""/>
        <figcaption><a href="https://zh.d2l.ai/chapter_attention-mechanisms/attention-cues.html#id4"> We ask ourselves to focus on the notebook. That's voluntary attention </a></figcaption>
    </figure>
</p>
</div>

## Query-Key-Value (QKV)

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/91e61a4d-4680-4273-88d3-8323855028ed" height="300" alt=""/>
    </figure>
</p>
</div>

When objects enter a machine eye, in our head, they will have a key (a short code), a value (e.g., their pixel values). Based on the machine brain's "voluntary attention", the brain will issue a query "what should I see if I want to work?". They query will be run through all objects' keys, and based on their similarity (or relavance), each object's value get assigned to a relavance score, then gets added up, and outputted as the combined "attention".  

More formally, the combined attention is

$$
\begin{gather*}
f(q, k1, v1, ...) = \sum_i \alpha(q, k_i) v_i
\end{gather*}
$$

where the attention weight $\alpha_i$ for the `ith` key value pair is:

$$
\begin{gather*}
\alpha(q, k_i) = \text{softmax}(a(q, k_i)) = \frac{exp(a(q, k_i))}{\sum_I exp(a(q, k_j))}
\end{gather*}
$$

Now, let's talk about how to calculate the attention score `a(q, k_i)`. There are two types: additive attention, and scaled dot-product attention.

### Additive (Bahdanau) Attention

When keys and the query have different lengths, we can use the **additive attention**. Additive attention projects keys and the **query into the same length** using two linear layers.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/0c9fcee6-c08b-4b8a-9132-5560575529a9" height="300" alt=""/>
    </figure>
</p>
</div>

$$
\begin{gather*}
a = W_v^T tanh(W_k k + W_q q)
\\
\alpha = softmax(a)
\end{gather*}
$$

The above can be implemented as a single multi-layer perceptron. Below is from the [seq2seq tutorial on PyTorch](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

- key vector `k` is `dk` long
- query vector `q` is `dq` long
- Say we have `h` as hidden dimension
- Learnable weight matrices $W_v$ `(h, 1)`, $W_k$ `(h, dk)`, $W_q$ `(h, q)` score how weighted queries and keys match with each other.

```python
from torch import nn
import torch

class BahdanauAttention(nn.Module):
    def __init__(self, key_size, query_size, hidden_size, dropout_p):
        super().__init__()
        self.Wk = nn.Linear(key_size, hidden_size, bias=False)
        self.Wq = nn.Linear(query_size, hidden_size, bias=False)
        self.Wv = nn.Linear(hidden_size, 1, bias=False) # a vector
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, queries, keys, values):
        """
        queries: (batch_size, query_num, query_size)
        keys: (batch_size，total_num_key_value_pairs， key_size)
        values: (batch_size，total_num_key_value_pairs， value_size)
        """
        # Project queries and keys onto the same hidden dim
        queries = self.Wq(queries)  # (batch_size, query_num, hidden_size)
        keys = self.Wk(keys)    # (batch_size，total_num_key_value_pairs，hidden_size)
        
        # Broadcasting to add queries and keys together
        queries = queries.unsqueeze(2)  # (batch_size, query_num, 1, hidden_size)
        keys = keys.unsqueeze(1)        # (batch_size, 1, total_num_key_value_pairs, hidden_size)
        features = queries + keys   # (batch_size, query_num, total_num_key_value_pairs, hidden_size)
        features = torch.tanh(features)

        scores = self.Wv(features)  # (batch_size, query_num, total_num_key_value_pairs, 1)
        scores = scores.squeeze(-1) # (batch_size, query_num, total_num_key_value_pairs)
        
        # Use masked_softmax here with a pre-designated length 
        self.attention_weights = nn.functional.softmax(scores)
        
        # torch.bmm is batch-matrix-multiplication
        # (batch_size, query_num, value_size), so we get all queries, weighted
        attention = torch.bmm(self.dropout(self.attention_weights), values)
        return attention

value_size = 2
key_size = 3
query_size = 4
hidden_size = 5
attention = BahdanauAttention(key_size=key_size, query_size=query_size, hidden_size=hidden_size, dropout_p=0.1)

batch_size = 1
query_num = 2
total_num_key_value_pairs = 3

torch.manual_seed(42)
queries = torch.rand((batch_size, query_num, query_size))
keys = torch.rand((batch_size, total_num_key_value_pairs, key_size))
values = torch.rand((batch_size,total_num_key_value_pairs, value_size))

attention(queries, keys, values)
```

### Scaled Dot-Product (Luong) Attention

When keys and queries do have the same length, dot-multiplying them together **is faster to give a "relavance" score**. Assume Queries is `num_queries x hidden_length (d)`, keys `key_pair_num x hidden_length`, values `key_pair_num x value_length`. Below, we denote the length of keys `hidden_length` as $d_k$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/6b3c7689-a533-4d79-aa81-841bb08b011d" height="300" alt=""/>
    </figure>
</p>
</div>

Note that if every pair of elements in keys and queries are independent with `[mean=0, var=1]`, their product $QK^T$ has a zero mean, and a variance `d`. We normalize this product and choose it to be our attention score `a`, so its variance is always 1.

$$
\begin{gather*}
a = \frac{QK^T}{\sqrt{d_k}}
\end{gather*}
$$

**Mask `M`** is applied before softmax, after calculating the attention score `a` (not shown in the illustration). In the attention is all you need paper, the [look-ahead mask is applied](./2022-03-26-deep-learning-transformer1-positional-encoding-and-masking.markdown/#look-ahead-mask) to make sure future 

The "attention is all you need" paper worded this point only with the basic intent, which I found confusing at the first time 😭

```
"We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections"
```

Then the attention weight is:

$$
\begin{gather*}
\alpha = softmax(\frac{QK^T}{\sqrt{d_k}} + M)V
\end{gather*}
$$

```python
class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

### Masked Softmax Operation

In real life applications, we might have a lot of input items, like words in an input sentence. Some of them are not very meaningful. Therefore, we can mask out the region with a pre-designated length to calcualate attention (softmax-value) on. To show the **effect**, below we mask out effective regions in each row.  

```python

def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # Set the masked out values (logits) to a large negative value so its softmax is close to zero 
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]]))

# see
tensor([[[1.0000, 0.0000, 0.0000, 0.0000],
         [0.4125, 0.3273, 0.2602, 0.0000]],

        [[0.5254, 0.4746, 0.0000, 0.0000],
         [0.3117, 0.2130, 0.1801, 0.2952]]])
```

## Visualization of Attention

One great feature about attention is its visibility. Below is an example from [the PyTorch NLP page](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

The input sentence is "il n est pas aussi grand que son pere".

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/08b15856-58ed-4ebb-afd5-c5db63ca3f67" height="300" alt=""/>
    </figure>
</p>
</div>

To interpret:

1. When outputting "he", most attention was given to "il", "n", "est"
2. When outputting "is", most attention was given to "aussi", "grand", "que" (which is interesting because `is` should be `est`)
3. When outputting "not", most attention was given to "aussi", "pas", "que"
4. The output "his father" focuses on "son père," which matches the intended translation.

## Bahdanau Encoder-Decoder Structure

In 2014, Bahdanau et al. proposed an encoder-decoder structure **on top of the additive attention**. To illustrate, we have a neural machine translation example (NMT): **translate French input "Jane visite l'Afrique en septembre" to English**. For attention pooling, we talked about scaled dot-product attention pooling and additive attention pooling in the previous sections.

- Encoder: we are using a [bi-directional RNN encoder](../2022/2022-03-15-deep-learning-rnn3-lstm.markdown) to generate embeddings of french sentences. Now, our input "Jane visite l'Afrique en septembre" will complete its forward and backward passes.
  - At each time `t`, the bidirectional RNN encoder outputs **a hidden state** $a^{(t)}$ (which is the key and value at the same time.)
  - $\alpha^{(t, t')}$: amount of attention output at time `t`, $y^{(t)}$ should put to hidden state at time `t'`, $a^{(t)}$
- Decoder we have **another single-drectional RNN decoder** to generate the word probabilities in the vocab space.
  - Here, we denote the hidden states as $s^{(t)}$. That's the **query**
  - Before outputting `<EOS>`, we assign a weight to several temporal neighbors in the input sequence at each time step.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/9bc9e775-cba6-413f-8787-bb87e3d027a0" height="300" alt=""/>
       </figure>
    </p>
</div>

In this case, we look at 3 neighbors and assign weights to them: $\alpha_1$, $\alpha_2$, $\alpha_3$. So, before outputting "Jane", we look at "Jane", "visite", "l'Afrique" at the same time. Note that this weighted sum of neighbors will enter the RNN cell as the **cell state**.

When we read **long** sentences, we have attention for short word segments before finishing the whole sentence. RNN networks's Bleu scores usually dips after a certain length. The attention mechanism, however, has much better performance.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/cc7c987a-367c-4fdb-89bc-453846091c76" height="200" alt=""/>
       </figure>
    </p>
</div>

**The process to learn the attention weight $\alpha^{(t, i)}$ is called "alignment"**. "Alignment" is to find the matching patterns between the input and the output. Specifically, alignment is learning the focus to put onto each encoder hidden state. This alignment model is said to **be "soft" so it allows back-propagation** and can be trained with the whole translation model

### Implementation

TODO: homework: what if we use scaled-dot product attention instead of the additive attention?

## References

[1] [Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. In Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015). https://arxiv.org/abs/1409.0473](https://arxiv.org/pdf/1409.0473)

[2] [Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhutdinov, R., Zemel, R., and Bengio, Y. 2015. Show, attend and tell: Neural image caption generation with visual attention. In Proceedings of the International Conference on Machine Learning (ICML).](https://arxiv.org/pdf/1502.03044)
