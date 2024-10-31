---
layout: post
title: Deep Learning - Hands-On Dinosour Name Generator Using RNN
date: '2022-03-21 13:19'
subtitle: 
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
    - Hands-on
---

## Character-Level Dinosour Name Generation

Build a character-level text generation model using an RNN.

- The vocabulary looks like:

```
{   0: '\n',
    1: 'a',
    2: 'b',
    3: 'c',
    4: 'd',
    5: 'e',
...
}
```

- Text corpus (which will be used as X, Y) looks like:

```
Aachenosaurus
Aardonyx
Abdallahsaurus
Abelisaurus
...
```

- At each step, output $y^{(t)}$ is fed back into the network as $x^{(t+1)}$

**The goal is to train the RNN to predict the next letter in the name, given a one-hot vector input.**

- So the labels are the list of characters that are one time-step ahead of the characters in the input `X`. For example, `Y[0]` contains the same value as `X[1]`
- So this is one example "dinarous", but spread out in multiple time steps?
- The RNN should predict a newline at the last letter, so add `ix_newline` to the end of the labels.

### Workflow

Training Phase:

1. We feed a sequence of input characters X into the RNN.
2. The model outputs a sequence of probability distributions $y_{hat}$, each over the entire vocabulary.
3. The true labels $Y$ are the next characters in the sequence, shifted by one time step from X, so $X(1) = Y(0)$
4. We compute the cross-entropy loss between $y_{hat}$ and $y$
5. This loss is used to perform gradient descent and update the model's parameters, enabling it to learn to predict the next character in a sequence.

Inference Phase (random sampling):

1. Given an initial character (or a special start token), the model generates the next character by predicting the most probable character that follows, by sampling from the probability distribution $y_{hat}$
    - The start token is a zero vector.
2. This predicted character is then used as the input for the next time step.
3. The sequence generation continues iteratively, each time feeding the previously generated character back into the model.
4. The sequence terminates when the model predicts a newline character '\n', indicating the end of the word (dinosaur name).

### Network Structure

Each cell is a regular RNN cell. This is inspired by [Andrej Karpathy's minimal example](https://gist.github.com/karpathy/d4dee566867f8291f086)

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/8c95c27d-9002-4fd3-9839-c234d8319f21" height="300" alt=""/>
    </figure>
</p>
</div>

## Music Generator

What's a value in music? **Informally, it could be a note. In music theory, you might need a chord (2 keys pressed at the same time)**

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/3578f6c1-1a63-44d9-81a9-f095bad52610" height="300" alt=""/>
        <figcaption>The Model Uses A Vanilla LSTM</figcaption>
    </figure>
</p>
</div>

### Training

 The music generation system will use 90 unique values. Our data is . In training, we use random snippets of 30 values taken from a much longer piece of music.

- Input X of shape `[m (batch_num), T_x(time), 90-one-hot-vector]` and labels Y of shape `(𝑇𝑦,𝑚,90)`, `T_y = T_x` and they are times. This makes it easier to be fed into LSTM?
- We feed $Y_{t-1}$ as $X_t$ into the LSTM.
- Training the model on random snippets of 30 values taken from a much longer piece of music.
  - The model is an LSTM with hidden states `C`, `a` that have 𝑛𝑎=64 dimensions.

In training, the entire input sequence 𝑥⟨1⟩,𝑥⟨2⟩,…,𝑥⟨𝑇𝑥⟩ is given in advance, then Keras has simple built-in functions to build the model. However, **for sequence generation, at test time you won't know all the values of 𝑥⟨𝑡⟩ in advance.** Instead, you'll generate them one at a time using 𝑥⟨𝑡⟩=𝑦⟨𝑡−1⟩ . That's why we are sharing weights among LSTM_cell, densor, reshaper. So, the function `djmodel()` will call the LSTM layer 𝑇𝑥 times using a for-loop.

- It is important that all 𝑇𝑥 copies have the same weights, then do back-prop later

```python

X, Y, n_values, indices_values, chords = load_music_utils('data/original_metheny.mid')

# number of dimensions for the hidden state of each LSTM cell.
n_a = 64
n_values = 90 # number of music values
reshaper = Reshape((1, n_values))                  # Reshaping a tensor to (1, n_values)
# By default, recurrent_activation='sigmoid'
LSTM_cell = LSTM(n_a, return_state = True)         # the hidden_state vectors a and c is `n_a`
densor = Dense(n_values, activation='softmax')     # TODO, what is this for?

# Each cell has the following schema: [X_{t}, a_{t-1}, c0_{t-1}] -> RESHAPE() -> LSTM() -> DENSE()
n_values = densor.units
n_a = LSTM_cell.units   # hidden state vector
X = Input(shape=(Tx, n_values)) 
Tx=30

# Define the initial hidden state a0 and initial cell state c0
a0 = Input(shape=(n_a,), name='a0')
c0 = Input(shape=(n_a,), name='c0')
a = a0
c = c0

outputs = []
for t in range(Tx):
    x =  X[:,t,:]
    x = reshaper(x)
    # Perform one step of the LSTM_cell
    _, a, c = LSTM_cell(x, initial_state=[a, c])
    # Apply densor to the hidden state output of LSTM_Cell
    out = densor(a)
    outputs.append(out)
    
# Step 3: Create model instance. Total params: 45,530
model = Model(inputs=[X, a0, c0], outputs=outputs)
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))
history = model.fit([X, a0, c0], list(Y), epochs=100, verbose = 0)
```

The dense layer is to convert the output of LSTM (64-vector) into 90-vector (output probability)

- 64 (input units) * 90 (output units) + + 90 (biases) = 5,850
- Remember that in LSTM, $y^(t) = softmax(a^(t))$

#### Keras Lessons

- In order to propagate a Keras tensor object X through one of these layers, use `layer_object()`.
  - For one input, use `layer_object(X)`
  - For more than one input, put the inputs in a list: `layer_object([X1,X2])`

- On `tf.keras.layers.LSTM_cell`: If a GPU is available and all the arguments to the layer meet the requirement of the cuDNN kernel (see below for details), the layer will use a fast cuDNN implementation when using the TensorFlow backend
  - "initial_state": List of initial state tensors to be passed to the first call of the cell (optional, None causes creation of zero-filled initial state tensors). Defaults to None.

- `layer.units` is an attribute that could be found in `LSTM`, `Dense` layers.
- `tf.keras.Input(shape=(Tx, n_values))`: symbolic tensor-like object that shows inputs and outputs clearly
  - exclude the batch size, as Keras handles batching automatically.

- Another way to reshape: create a reshape layer

```python
reshaper = Reshape((1, n_values))                  # Used in Step 2.B of djmodel(), below
reshaper(X)
```

- Instantiate a `model`. One is through the Functional API, which is flexible and allows chained layer calls

```python
model = Model(inputs=[input_x, initial_hidden_state, initial_cell_state], outputs=the_outputs)
```

- Pass actual inputs into the model:

```python
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))
# X, a0, c0 corresponds to the Input symbols in the model, output corresponds to the outputs variable.
history = model.fit([X, a0, c0], list(Y), epochs=100, verbose = 0)
```

Loss:
<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/3c329b7f-3995-45e9-9fa2-3c552de440d1" height="200" alt=""/>
    </figure>
</p>
</div>

### Inferencing

```python
def music_inference_model(LSTM_cell, densor, Ty=100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    """
    
    n_values = densor.units
    # Get the number of the hidden state vector
    n_a = LSTM_cell.units
    
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_values))
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    outputs = []
    for t in range(Ty):
        # Perform one step of LSTM_cell. Use "x", not "x0" (≈1 line)
        _, a, c = LSTM_cell(x, initial_state=[a, c])
        # Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
        out = densor(a)
        # Append the prediction "out" to "outputs". out.shape = (None, 90) (≈1 line)
        outputs.append(out)
 
        x = tf.math.argmax(out, axis=-1)
        x = tf.one_hot(x, depth=n_values)
        # Use RepeatVector(1) to convert x into a tensor with shape=(None, 1, 90)
        x = RepeatVector(1)(x)
        
    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    return inference_model

x_initializer = np.zeros((1, 1, n_values))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))
```

- `RepeatVector(1)(x)` inputs `(batch_size, features)` and `(batch_size, 1, features)`
- "Your results may likely differ because Keras' results are not completely predictable. "", why?
  - Sampling operations
  - Dropout Layers
