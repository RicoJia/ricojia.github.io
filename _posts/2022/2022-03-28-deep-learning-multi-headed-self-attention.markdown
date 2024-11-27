---
layout: post
title: Deep Learning - Multi-Head and Self Attention
date: '2022-03-27 13:19'
subtitle: Multi-Head Attention, Self Attention, Comparison of Self Attention Against CNN, RNN
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Multi-Head Attention

To learn a richer set of behaviors, we can instantiate multiple attentions jointly given the **same set of queries, keys, and values**. Specifically, we are able to capture various long-range and short range dependencies.

The Process is:

1. Linearly transform `q`, `k`, `v` into `q'`, `k'`, `v'` **so that they all have the same hidden dimension `hidden_size`**
    - In the meantime, it adds learnability for the non-linear decision landscape.
1. Split `q'`, `k'`, `v'` into heads: `h1_q`, `h1_k`, `h1_v`, `h2_q`, `h2_k`, `h2_v`. A head is a part of the overall `q'`, `k'`, `v'`.
1. The attention module is [additive or scaled-product attention pooling](./2022-03-27-deep-learning-attention-mechanism.markdown). The attention module does not have any learnable parameters. **They run on each head in parallel**.
    - For each head $i$, attention is calculated based on its unique $W_i^Q Q$, $W_i^K K$ , $W_i^V V$
    - In the "Attention is All You Need" paper, 8 heads were used.
1. All `h` are concatenated
3. The concatenated head is transformed into a shorter embedding through a dense layer, `Wo`

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/152c1a01-8123-449c-8f50-09b1cc6c967a" height="600" alt=""/>
       </figure>
    </p>
</div>

- I'm omitting the lenght masking part in the illustration. In reality we add it to focus on the generally relavant segment of the input sentence.

The reason why multi-headed attention works so well is: **each input word will evolve into embeddings (i.e., key and value). Then the embeddings are divided into heads, where each head could represent a different meaning of the word. So, attention weights are given to different meanings of each individual word, based on the sub-vectors of other words.** Finally, the overall weighed-attention sub-vectors are calculated, concatenated together, transformed into an overall embedding.

One might notice that the linear transformations and the final attention pooling action (just the dot-product part) share the same weights across heads. This keeps the model small, yet still appears to be effective in real life.

- In the code below, the embedding is of length `hidden_size`

$$
\begin{gather*}
W_o [h_1, ... h_n]
\end{gather*}
$$

Now, let's enjoy the code

```python
"""
Notes:
- Lazy*: borrow the nice input channel inference feature from TensorFlow
    e.g., torch.nn.LazyLinear(out_features=hidden_size, bias=False)
"""
import torch
import math

class DotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self,queries, keys, values):
        """ softmax(q k^T/sqrt(d_q)) * v

        Args:
            queries: (batch_size * num_heads, num_queries, head_dim)
            keys: (batch_size * num_heads, num_kv, head_dim)
            values: (batch_size * num_heads, num_kv, head_dim)
        Returns: 
            Output attention (batch_size * num_heads, num_queries, head_dim)      
        """
        head_dim = queries.shape[-1]
        # we assume each row in queries is indenpendent from each column in keys.transpose()
        # So its raw production has a standard deviation of torch.sqrt(head_dim). This is normalization
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(head_dim) #(num_heads, num_queries,num_kv)
        # TODO: should apply masked output
        self.attention_weights = torch.nn.functional.softmax(scores,  dim=-1) #(num_heads, num_queries,num_kv)
        output = torch.bmm(self.dropout(self.attention_weights), values)    #(num_heads, num_queries,head_dim)
        return output
        
def transpose_qkv(X, num_heads):
    """Shape transform: 
    (batch_size, num_kv/num_queries, hidden_size) -> 
    (batch_size, num_kv/num_queries, num_heads, hidden_size//num_heads) ->
    (batch_size, num_heads, num_kv/num_queries, hidden_size//num_heads) ->
    (batch_size * num_heads, num_kv/num_queries, hidden_size//num_heads) ->
    
    This is to make heads part of the batch for parallel computation
    """
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """Reverse the operation of transpose_qkv."""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiheadedAttention(torch.nn.Module):
    def __init__(self, hidden_size, output_size, num_heads):
        super().__init__()
        # Code up an attention, then instantiate them multiple times?
        self.Wq = torch.nn.LazyLinear(out_features=hidden_size, bias=False)
        self.Wk = torch.nn.LazyLinear(out_features=hidden_size, bias=False)
        self.Wv = torch.nn.LazyLinear(out_features=hidden_size, bias=False)
        self.Wo = torch.nn.LazyLinear(out_features=output_size, bias=False)
        self.attention = DotProductAttention()
        self.num_heads = num_heads
    def forward(self, queries, keys, values):
        """
        queries: (batch_size, query_num, query_size)
        keys: (batch_size，total_num_key_value_pairs， key_size)
        values: (batch_size，total_num_key_value_pairs， value_size)
        """
        q_prime = self.Wq(queries) #(batch_size, num_queries, hidden_size)
        k_prime = self.Wk(keys) #(batch_size, num_kv, hidden_size)
        v_prime = self.Wv(values) #(batch_size, num_kv, hidden_size)
        
        queries = transpose_qkv(q_prime, self.num_heads)  #(batch_size * num_heads, num_queries, hidden_size//num_heads)
        keys = transpose_qkv(k_prime, self.num_heads)   #(batch_size * num_heads, num_kv, hidden_size//num_heads)
        values = transpose_qkv(v_prime, self.num_heads) #(batch_size * num_heads, num_kv, hidden_size//num_heads) 
        
        output_heads = self.attention(queries, keys, values) #(batch_size * num_heads, num_queries, hidden_size//num_heads)
        output_concat = transpose_output(output_heads, self.num_heads)  #(batch_size, num_queries, hidden_size)
        output = self.Wo(output_concat) #(batch_size, num_queries, output_size)
        return output

batch_size = 1
num_kv = 5
num_queries = 3
input_dim = 4
num_hiddens = 32
num_heads = 2
head_dim = num_hiddens // num_heads  # 8
output_size = 8
dropout = 0.1

# Set random seed for reproducibility
torch.manual_seed(0)

# Generate random queries, keys, and values
queries = torch.randn(batch_size, num_queries, input_dim)
keys = torch.randn(batch_size, num_kv, input_dim)
values = torch.randn(batch_size, num_kv, input_dim)
# Instantiate MultiHeadAttention
multi_head_attn = MultiheadedAttention(hidden_size=num_hiddens, output_size=output_size, num_heads=num_heads)
output = multi_head_attn(queries, keys, values)
```

## Self Attention

when key, value, and query come from the same set of inputs, they are called "self-attention" [1]. We **also want to make sure the output has the same dimension as the inputs**. Since value and queries are the same, this is equivalent to having `num_queries` input words, and having `num_queries` output words

Now let's illustrate with some code

```python
num_hiddens, num_heads = 100, 5
attention = MultiheadedAttention(hidden_size=num_hiddens, output_size=num_hiddens, num_heads=num_heads)
attention.eval()
batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
output = attention(X, X, X) #(batch_size, num_queries, num_hiddens)
print(attention)
```

### Comparing CNN, RNN, and Self-Attention

Saywe are given an `n` input tokens. They are a `nxd` vector. We are outputting a sequence of `dxn` as well. We compare:

- Time complexity
- Sequential Operations: number of actions which takes place in sequence. They are bottlenecks of parallel computations
- Maximum Path Lengths: the length (or number of layers) needed to allow for any input element to be considered in any output sequence.
  - For example, with 1 layer CNN, the first input element is considered within the first output element, but not the subsequent ones as the kernel moves forward. We need to have more layers so that the first element is considered.
  - A shorter path between any combination of sequence positions makes learning long-range dependencies easier

For CNN:

- Input and output channels are `d`; kernel size is `k`
- Time complexity: $O(nd^2k)$ because we need to go over all elements in the input and output filters
- Sequential operations: we need to calculate layer by layer, but we know that beforehand, so $O(1)$
- Maximum Path length: (receptive field size?) is $O(n/k)$, For example, x1, x5 are within the receptive fields of CNN

For RNN:

- Say we have 1 layer, since we are outputting with the same dimension, the hidden state dimension is `d` as well.
- Time Complexity: weight matrices are `dxd`. In total, $O(nd^2)$
- Sequential Complexity: $O(n)$
- Maximum path length: $O(n)$ as we need to finish the entire $n$ timesteps so the last output sequence can technically see the first input element.

For Self Attention:

- Time Complexity: weight matrices are `nxd`. In total, $O(n^2d)$
- Sequential Complexity: $O(1)$: we need to do linear transform, concatenate, and dense layer.
- Maximum path length: $O(1)$ as the single operation is able to consider all input elements.

**So, both CNN and self attention has a low number of sequential operations and are highly parallelizable. However, self attention will suffer from higher complexity when input sequence is long.**

## References

[1] [Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems (pp. 5998–6008).](https://arxiv.org/pdf/1706.03762)
