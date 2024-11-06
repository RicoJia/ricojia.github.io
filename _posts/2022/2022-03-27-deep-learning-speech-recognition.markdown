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

When we read **long** sentences, we have attention for short word segments before finishing the whole sentence. RNN networks's Bleu scores usually dips after a certain length. The attention mechanism, however, has much better performance.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/cc7c987a-367c-4fdb-89bc-453846091c76" height="200" alt=""/>
       </figure>
    </p>
</div>

The intuition is as follows. Say we are using a [bi-directional RNN encoder](../2022/2022-03-15-deep-learning-rnn3-lstm.markdown) to generate embeddings of french sentences. Now, our input "Jane visite l'Afrique en septembre" will complete its forward and backward passes.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/7aeab525-f79c-4331-a92e-9b97b8f74011" height="300" alt=""/>
       </figure>
    </p>
</div>

After that, we will have **another single-drectional RNN decoder** to generate the word probabilities in the vocab space. Here, we denote the hidden states as $s^{(t)}$. Before outputting `<EOS>`, we assign a weight to several temporal neighbors in the input sequence at each time step (TODO is this each time?). In this case, we look at 3 neighbors and assign weights to them: $\alpha_1$, $\alpha_2$, $\alpha_3$. So, before outputting "Jane", we look at "Jane", "visite", "l'Afrique" at the same time. Note that this weighted sum of neighbors will enter the RNN cell as the **cell state**.

## Attention Construct

Ok, now let's formalize the attention construct. I'm going to formalize the notations first:

- At each time `t`, the bidirectional RNN encoder outputs **a hidden state** $a^{(t)}$
- $\alpha^{(t, t')}$: amount of attention output at time `t`, $y^{(t)}$ should put to hidden state at time `t'`, $a^{(t)}$
- $e^{(t, i)}$: scalar **attention score** for each hidden state `i`

To learn $\alpha^{(t, t')}$, we need to look at all hidden states from all encoder hiden states $a^{(t)}$. Assuming the encoder timesteps are $[0, T]$, we calculate a scalar **attention score**, $e^{(t, i)}$ for each hidden state `i` that considers the last decoder hidden state $s^{(t-1)}$ and the encoder hidden state $h^{(i)}$

To calculate the scalar **attention score**, $e^{(t, i)}$, we have an **alignment model**. "Alignment" is to find the matching patterns between the input and the output. Specifically, alignment is learning the focus to put onto each encoder hidden state. This alignment model is said to be "soft" so it allows back-propagation and can be trained with the whole translation model [1]. Here, let's look at the **Bahdanau attention (additive attention), the OG attention in 2015**.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/dc402b1e-770a-420a-be6c-66d75d566a0e" height="100" alt=""/>
       </figure>
    </p>
</div>

$$
\begin{gather*}
e^{(t, i)} = v^T tanh(W_s s^{(t-1)} + W_h h^{(i)})  \\
a^{(t, i)} = \frac{exp(e^{(t, i)})}{\sum_T exp(e^{(t, i)})} = softmax(e^{(t, i)}) \text{(across all T)} \\
c^{(t)} = \sum_T a^{(t, i)} h^{(i)}
\end{gather*}
$$

The above can be implemented as a single multi-layer perceptron. Below is from the [seq2seq tutorial on PyTorch](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

```python
import torch
from torch import nn

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        # we are assuming the encoder and decoder have the same size
        # wa * a (encoder) + U * h (decoder)
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1) # a vector
    def forward(self, query, key):
        e = self.Va(self.Wa(query) + self.Ua(key))
        # Dims?
        e = e.squeeze(2).unsqueeze(1)
        weights = nn.functional.softmax(e, dim=-1)
        context = torch.bmm(weights, key) # weights * query, torch.bmm is batch-matrix-multiplication
        return weights, context
```

Important notes:

- Do $c^{t}$, $h_i$, $s^{(t)}$ have the same dimensions? Answer: $c^{t}$ and $h_i$ do, $s^{(t)}$ could be different

  - $h_i$ dimensions are from the encoder architecture. It can be different from $s^{(t)}$
  - $c^{t}$ is a weighted sum of $h_i$ across $i$, so their dimensions are the same
  - $s^{(t)}$ the hidden state of the decoder could have a different dimension

### Query-Key-Value (QKV)

Imagine we are consulting a dictionary about a question.

- A query is like a question "what's the most important part to learn"? In Bahdanau Attention, the query is the decoder hidden state
- A key is the index of the information you find in the dictionary tells you. In Bahdanau Attention, the key is a spific encoder hidden state
- A value is the relevant information you find for your key. In Bahdanau Attention, the value is the final context vector (TODO?)

I'm using the dictionary analogy because when figuring out the context, we need to ask "what's weight of each encoder's hidden state given my current decoder hidden state".

Bahdanau Attention is a "Cross Attention". In transformers, we have "self attention"

### Visualization of Attention

One great feature about attention is its visibility. Below is an example from [the PyTorch NLP page](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

The input sentence is "il n est pas aussi grand que son pere".

![sphx_glr_seq2seq_translation_tutorial_003](https://github.com/user-attachments/assets/08b15856-58ed-4ebb-afd5-c5db63ca3f67)

To interpret:

1. When outputting "he", most attention was given to "il", "n", "est"
2. When outputting "is", most attention was given to "aussi", "grand", "que" (which is interesting because `is` should be `est`)
3. When outputting "not", most attention was given to "aussi", "pas", "que"
4. The output "his father" focuses on "son père," which matches the intended translation.

## References

[1] [Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. In Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015). https://arxiv.org/abs/1409.0473](https://arxiv.org/pdf/1409.0473)

[2] [Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhutdinov, R., Zemel, R., and Bengio, Y. 2015. Show, attend and tell: Neural image caption generation with visual attention. In Proceedings of the International Conference on Machine Learning (ICML).](https://arxiv.org/pdf/1502.03044)
