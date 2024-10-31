---
layout: post
title: Deep Learning - RNN
date: '2022-03-09 13:19'
subtitle: Sequence Models, RNN Architectures
comments: true
header-img: "img/home-bg-art.jpg"
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Sequence Models

Some common sequence models include: DNA sequencing, audio clips, sentiment classification, etc. Another example is name indexing, where names in news for a past period of time will be searched so they can be indexed and searched appropriately.

The first step to NLP is to build a dictionary, $X$. Say we have the most common 10000 English words, we can use one-hot encoding to represent a word, and the word can be indexed as $x^{i}$. If we see a word that's not in the dictionary, the word is indexed `<UNK>` as "unknown".

A fully connected network / CNN doesn't work well for sequence data: 1. The sequence could be of arbitrary lengths 2. There's no weight sharing in these sequence models.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/b8ad0eb3-25bf-4dea-9e6f-cc6d6a903e06" height="200" alt=""/>
    </figure>
</p>
</div>

## RNN Architecture & Forward Propagation

The word "recurrent" means "appearing repeatedly". In an RNN, we have **hidden states** $a^{(t)}$, sequential inputs $x^{(t)}$, output $\hat{y}^{(t)}$. Each superscript $i$ represents timestamp, e.g., $a^1$ means a at time $1$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/de82d933-9400-46ce-83cc-bc5ea26d51a8" height="300" alt=""/>
    </figure>
</p>
</div>

$$
\begin{gather*}
a^{t} = g_0(W_{aa} a^{(t-1)} + W_{ax} x^{(t)} + b_a)
\\
\hat{y}^{i} = g_1(W_{ya} a^{(t)} + b_y)
\end{gather*}
$$

- Dimensions
  - Batch is usually dimension 1
  - $a^{0}$ is usually zero or randomly generated values. **a_0 do not need to be of the same length as x**.
  - $a^{(t)}$ are stacked on top of $x$
  - So $a$ could be $[5, 10]$, x could be $[3,10]$, with a batch size being `10`
- $g_0$ could be tanh (more common) or relu, $g_1$ could be sigmoid.
  - `output = softmax(V * (W*output_{t-1} + U*x_{t}))`
- $W_{ax}$ is a matrix that "generates a-like vectors, and takes in an x-like vector". Same notation for $W_{aa}$

We can simplify this notation into:

$$
\begin{gather*}
W_{a} = [W_{aa}, W_{ax}]
\\

a^{t} = g_0(W_{a}[a^{t-1}, x^{t}]^T + b_a)
\end{gather*}
$$

**RNN works best with local context (recent words).**

### Notations

- Superscript $(i)$ denotes an object associated with the $i^{th}$ example.
- Superscript $[l]$ denotes an object associated with the $l^{th}$ layer.
- Superscript $\langle t \rangle$ denotes an object at the $t^{th}$ time
step.
- Subscript $i$ denotes the $i^{th}$ entry of a vector.

$$
a_{i}^{[i][l](t)} => a_5^{[2](3)<4>}
$$

#### Architectures

- Above is **many to many**, meaning you have multiple inputs and have multiple outputs.
- There's another type of **many to many**, where we have different lengths of outputs and inputs e.g., machine translation

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/b4963c3e-c98b-492f-9b58-6ff95a2d82a9" height="200" alt=""/>
    </figure>
</p>
</div>

- We also have **many to one**, meaning you have multiple inputs and one output, like sentiment analysis

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/c946f898-eaf5-4b04-adae-cc9c08c2ab76" height="200" alt=""/>
    </figure>
</p>
</div>

- We also have **one to many**, like music generation that takes in one prompt input and generates multiple notes.  

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/0a19bd68-5bc6-4163-bee9-51e3a7a82684" height="200" alt=""/>
    </figure>
</p>
</div>

Disadvantages:

- Inputs in the early sequence are not influenced by inputs later in the sequence. One example is "Teddy bear is on sale", Teddy is not a name, while many times it is.

## Language Modelling

A language model takes in a sequence of words and outputs new sequence like in speech recognition and machine translation. The output is the **most probable** output. E.g.,

```
English input: I love Turkish coffee
Turkish output: Ben türkçe kahvesi seviyorum
```

Input: Tokenize the sentence into one-hot vectors. Add an end-of-sentence token `<EOS>` to be explicit. If there's a word that's not in the vocabulary, use an unknown token to represent that `<UNK>`. Before outputing probablities, we need `softmax` to normalize.

This is equivalent to a Markov Decision Process. Each predition $\hat{y}^{(t)}$ is $p(y^{(t)} \| y^{(0)}, y^{(1)} ... y^{(t-1)})$. The whole sentence's probability is $p(y^{(0)}, y^{(1)} ... y^{(t-1)}, y^{(t)})$

 Let's walk through an example with arbitrarily-assigned probabilities.

1. I (one-hot vector is [0, , ... 1, ... 0]) -> $P(y_0 = Ben)$ = 0.4 (so take 'Ben' as output)
2. love -> P(seviyor) = 0.00003, P(seviyorum) = 0.00001
    1. P(seviyor \| Ben) = 1e-10, $P(\text{seviyorum} \| y_0 = \text{Ben}) = 0.4$
    2. So take the most probable sequence 'Ben seviyorum' as output.
3. `Turkish` -> $P(\text{Türkçe}) = 0.5$
    1. $P(y_2 = \text{Türkçe} \| y_0 = \text{Ben}, y_1 = \text{seviyorum}) = 8e^{-10}$, $P(y_1 = \text{Türkçe} \| y_0 = \text{Ben}, y_2 = \text{seviyorum}) = 6e^{-7}$
    2. So take 'Ben Türkçe seviyorum' as output
4. `Coffee` -> `P(Kahve) = 0.7, P(Kahvesi) = 0.2`:
    1. $P(y_3 = \text{Kahve} \| y_0 = \text{Ben}, y_1 = \text{Türkçe}, y_2 = \text{seviyorum}) = 7e^{-15}$
    1. ...
    1. $P(y_2 = \text{Kahvesi} \| y_0 = \text{Ben}, y_1 = \text{Türkçe}, y_3 = \text{seviyorum}) = 9e^{-9}$
    1. The most probable sequence is `Ben türkçe kahvesi seviyorum` as the output

- The sentence `Türkçe kahvesi seviyorum` has a probability of $9e^{-9}$ and is the highest among all sentences.

**One note is the above example is NOT how modern machine translation works using Neural Machine Translation (NMT). It's a simplification of word-by-word translation.**

- Modern NMTs, expecially transformers, do not rely on Markov Assumptions and consider not just previous words, but the entire input sequence.

- Probablities like $P(y_2 = \text{Türkçe} \| y_0 = \text{Ben}, y_1 = \text{seviyorum})$ are computed by complex neural networks to  based on learned representations of both the source and target languages.

`RNN` can have exploding/diminising gradient as well. For explosion, do gradient clipping. for diminishing, can try 1. weight init, 2. use relu instead of sigmoid 3. other RNNs: LSTM, GRU.

#### Training

The training set is a large corpus of English -> Turkish text.
Loss function is:

$$
\begin{gather*}
L(y^{(t)}, \hat{y^{(t)}}) = -\sum_i y_i^{(t)} log (\hat{y}^{(t)})
\\
L = -\sum_t L(y^{(t)}, \hat{y^{(t)}})
\end{gather*}
$$

Where $i$ is the dimension of one-hot vectors, and $t$ is time.

### Sampling Novel Sequences

After training a model, the model should have learned the conditional probability distribution $P(y_t \| y_1, ... y_{t-1})$. we can informally get a sense of what the model learned by sampling novel sequences. For example, our vocabulary is `['I', 'you', 'love', 'coffee', 'apple']`

1. At `t=0`
    1. Choose start tokens $a^{(0)} = 0, x^{(0)} = 0$ (Start-Of-Sequence `<SOS>` token),
    2. Generate hidden state $a^{(0)} = W_{ax} x^{(0)} + b_a$
    3. Generate distribution $y^{(0)} = softmax(W_{ya}a^{0} + b_y) = [0.6, 0.2, 0.05, 0.05, 0.1]$.
    2. Use a sampling strategy from the distribution $y^{(0)}$
        - **Greedy sampling**: choose the token with the highest probability. You might miss less probable but plausible sequences.
        - **Stochastic sampling**: treat $y^{(0)}$ as a  **categorical distribution** , then randomly draw a sample from it. one can use `np.random.choice()`.
        - In this step, we draw `'I'` and add it to the sequence
2. At `t=1`
    1. Generate hidden state $a^{(1)} = W_{ax} x^{(1)} + b_a$
    2. Generate distribution $y^{(1)} = softmax(W_{ya}a^{1} + b_y) = [0.2, 0.2, 0.4, 0.1, 0.1]$.
    3. After stochastic sampling, we draw 'love' and add it to the sequence

...

### Character-Level RNN

Above is "word-level" RNN. If inputs are charaters `[a-z, A-Z, 0-9, ...]`, you wouldn't have to worry about `<UKN>` tokens. But one disadvantage is your output sequence is much longer so the **long-range dependencies** might be missed. This is more computationally expensive.

## BackPropagation Through Time (BPTT)

RNN implementation

- $nx$ is used here to denote the number of units in a single time step of a single training example
- $Tx$ will denote the number of timesteps in the longest sequence.
- stack 20 **columns** of 𝑥(𝑖) examples

It's worth noting that:

$$
\begin{gather*}
tanh'(x) = (1-tanh^2(x))
\\
\sigma'(x) = \sigma(x) (1 - \sigma(x))
\end{gather*}
$$

So:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/cde79643-c3b4-487c-b572-24d7c86061d9" height="300" alt=""/>
    </figure>
</p>
</div>

$$
\begin{gather*}
\begin{align}
\displaystyle a^{\langle t \rangle} &= \tanh(W_{ax} x^{\langle t \rangle} + W_{aa} a^{\langle t-1 \rangle} + b_{a})\tag{-} \\[8pt]
\displaystyle \frac{\partial \tanh(x)} {\partial x} &= 1 - \tanh^2(x) \tag{-} \\[8pt]
\displaystyle {dtanh} &= da_{next} * ( 1 - \tanh^2(W_{ax}x^{\langle t \rangle}+W_{aa} a^{\langle t-1 \rangle} + b_{a})) \tag{0} \\[8pt]
\displaystyle  {dW_{ax}} &= dtanh \cdot x^{\langle t \rangle T}\tag{1} \\[8pt]
\displaystyle dW_{aa} &= dtanh \cdot a^{\langle t-1 \rangle T}\tag{2} \\[8pt]
\displaystyle db_a& = \sum_{batch}dtanh\tag{3} \\[8pt]
\displaystyle dx^{\langle t \rangle} &= { W_{ax}}^T \cdot dtanh\tag{4} \\[8pt]
\displaystyle da_{prev} &= { W_{aa}}^T \cdot dtanh\tag{5}
\end{align}
\end{gather*}
$$

Note that we need to accumalate hidden input gradient by

$$
\begin{gather*}
da = da_{loss[t]} + d{a_prev}
\end{gather*}
$$

- $da_loss[t]$ is the hidden state gradient from the immediate loss. So **a keypoint of BPTT** is consdering both the next time gradient and the immediate loss for hidden state gradient
- loss only considers the current timestamp output, and can use binary cross entropy loss.

```Python
for t in reversed(range(T_x)):
    # Compute the total gradient at time step t
    da_next = da[:, :, t] + da_prevt
    gradients = rnn_cell_backward(da_next, caches[t])
    dxt = gradients["dxt"]
    da_prevt = gradients["da_prev"]
    dWax += gradients["dWax"]
    dWaa += gradients["dWaa"]
    dba += gradients["dba"]
    dx[:, :, t] = dxt
```
