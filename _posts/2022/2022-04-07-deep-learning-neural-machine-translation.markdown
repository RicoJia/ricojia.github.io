---
layout: post
title: Deep Learning - Neural Machine Translation
date: '2022-04-07 13:19'
subtitle: Hands-On Attention Project
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Introduction And Data Preparation

The goal of the project is experimenting with date translations, i.e., ("25th of June, 2009") into machine-readable dates ("2009-06-25"). We need to truncate data if necessary

- Set max input length `Tx` to 30 char long
- Set max input length `Ty` to 10 char long

The code for getting the data is:

```python
"""
human_vocab: {' ': 0, '.': 1, '/': 2 ... 36}
machine_vocab: {'-': 0, '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10}
- X becomes [4, 3, ...] 30-d vector, where each character is the index in human_vocab that the element is mapepd to
- Y is the index of character in machine-vocab (a 10-d vector)
- Xoh: one-hot representation of X (30x37)
- Yoh: one-hot representation of Y (10x11)

Eventually, we want:
- Source date: 9 may 1998
- Target date: 1998-05-09
"""

Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
print(machine_vocab)

index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
```

## Model

When we read in English, we put focus on certain "important parts". The model we are using is a **Global Soft Model**, within an encoder-decoder framework.. In this attention mechanism, there are:

- A pre-attention **bi-directional** LSTM going through the entire `Tx` sequence
- A post-attention LSTM going through the **global** `Ty` output sequence. It passes cell state `C` and hidden state `S` from one timestep to the next.

**Specific to this model's post-attention LSTM, we only take hidden state `h` and cell state `c`.** In text generation, the post-attention LSTM would take hidden state `h` and the previous output `y_(t-1)`. This is because in language generation, adjacent chars have a strong dependency. But in dates YYYY-MM-DD, there isn't such a strong dependency.

- TODO: recreate the structure of context diagram (0.5h)

2. Compute energy $e^{(t, t')}$ as a function of the post-attention hidden state $s^{(t-1)}$ and pre-attention hidden state $a^{(t')}$. **$e^{(t, t')}$ is the attention $y^{(t)}$ should pay to $a^{(t')}$**.

- $s^{(t-1)}$ and $a^{(t')}$ are fed into a dense layer to get $e^{(t, t')}$. Then, $e^{(t, t')}$ gets into a softmax layer to compute $\alpha^{(t, t')}$
- Context

$$
context = \sum_{t'}^{T_x} \alpha^{(t, t')} a^{t'}
$$

- TODO: more explanation on `RepeatVector` copy $s^{(t-1)}$ `T_x` times

```python
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: one_step_attention
def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attention) LSTM cell
    """

    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    m, Tx, n_a2 = a.shape
    m, n_s = s_prev.shape
    # we MUST reuse the same repeator object
    # s_prev = RepeatVector(Tx)(s_prev)
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    # For grading purposes, please list 'a' first and 's_prev' second, in this order.

    #     concat = Concatenate(axis = -1)([a, s_prev])
    concat = concatenator([a, s_prev])

    # concat.shape = TensorShape([10, 30, 128])
    #     print(f"concat.shape: {concat.shape, Ty}")
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e.
    # rememebr, e is of Ty because it encompasses all output timesteps
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
    energies = densor2(e)
    print(f"e.shape, energies.shape: {e.shape, energies.shape}")
    # See TensorShape([10, 30, 10]), TensorShape([10, 30, 1])
    # Rico: part of the reason why tf is not popular is because the layer sizes are partially
    # inferred, you don't know the full dimensions just from the model definition

    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a", in this order, to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = dotor([alphas, a])

    return context
```

- `modelf`:

```python
n_a = 32 # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
n_s = 64 # number of units for the post-attention LSTM's hidden state "s"

post_activation_LSTM_cell = LSTM(n_s, return_state = True) # Please do not modify this global variable.
output_layer = Dense(len(machine_vocab), activation=softmax)

def modelf(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """

    # Define the inputs of your model with a shape (Tx, human_vocab_size)
    # Define s0 (initial hidden state) and c0 (initial cell state)
    # for the decoder LSTM with shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    # initial hidden state
    s0 = Input(shape=(n_s,), name='s0')
    # initial cell state
    c0 = Input(shape=(n_s,), name='c0')
    # hidden state
    s = s0
    # cell state
    c = c0

    # Initialize empty list of outputs
    outputs = []

    ### START CODE HERE ###

    # Step 1: Define your pre-attention Bi-LSTM. (≈ 1 line)
    a = Bidirectional(LSTM(units=n_a, return_sequences=True))(X)

    # Step 2: Iterate for Ty steps
    for t in range(Ty):

        context = one_step_attention(a, s)

        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector. (≈ 1 line)
        # Don't forget to pass: initial_state = [hidden state, cell state]
        # Remember: s = hidden state, c = cell state
        _, s, c = post_activation_LSTM_cell(context, initial_state=[s,c])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)

        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)

    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs=[X, s0, c0], outputs=outputs)

    ### END CODE HERE ###

    return model
```

### Learned Result

We can plot the "attention map" of a given input: `"Tuesday 09 Oct 1993"`. We can see that at each output character (or time step), attentions (weights) are given to the right input characters.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/ea04d38d-d92c-4f91-a518-ea1353128884" height="300" alt=""/>
       </figure>
    </p>
</div>
