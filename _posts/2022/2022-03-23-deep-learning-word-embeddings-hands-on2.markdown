---
layout: post
title: Deep Learning - Word Emojifier Using Dense and LSTM Layers
date: '2022-03-23 13:19'
subtitle: Emojifier
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
    - Hands-on
---

## Introduction

When using word vectors, you'll see that even if your training set explicitly relates only a few words to a particular emoji, your algorithm will be able to generalize and associate additional words in the test set to the same emoji?

This works even if those additional words don't even appear in the training set.
This allows you to build an accurate classifier mapping from sentences to emojis, even using a small training set.

1. In this exercise, you'll start with a baseline model (Emojifier-V1) using word embeddings.
2. Then you will build a more sophisticated model (Emojifier-V2) that further incorporates an LSTM.

- Create an embedding layer in Keras with pre-trained word vectors
- Explain the advantages and disadvantages of the GloVe algorithm
- Build a sentiment classifier using word embeddings
- Build and train a more sophisticated classifier using an LSTM

You have a tiny dataset (X, Y) where:

- X contains 127 sentences (strings).
- Y contains an integer label between 0 and 4 corresponding to an emoji for each sentence.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/8e426be3-594a-4351-a632-814f50e3d8a4" height="300" alt=""/>
        <figcaption>Training Data Overview</figcaption>
    </figure>
</p>
</div>

## Emojifier-V1

The model is a single linear layer + softmax.

$$ z^{(i)} = Wavg^{(i)} + b$$

$$ a^{(i)} = softmax(z^{(i)})$$

$$ \mathcal{L}^{(i)} = - \sum_{k = 0}^{n_y - 1} Y_{oh,k}^{(i)} * log(a^{(i)}_k)$$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/51c7341e-67c2-42d4-b0e7-4882a73d3e7b" height="300" alt=""/>
        <figcaption>Model Structure</figcaption>
    </figure>
</p>
</div>

```python
# X_train = ["I am so excited to see you after so long", ...]
# label_to_emoji(Y_train[idx]) = [0, 3, ...]. There are 5 emoji labels in total.
maxLen = len(max(X_train, key=lambda x: len(x.split())).split())

Y_oh_train = convert_to_one_hot(Y_train, C = 5)
Y_oh_test = convert_to_one_hot(Y_test, C = 5)

# Validating that we are getting the input data right
idx = 50
print(f"Sentence '{X_train[idx]}' has label index {Y_train[idx]}, which is emoji {label_to_emoji(Y_train[idx])}", )
print(f"Label index {Y_train[idx]} in one-hot encoding format is {Y_oh_train[idx]}")

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word and averages its value into a single vector encoding the meaning of the sentence.

    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (J,), where J can be any number
    """
    any_word = next(iter(word_to_vec_map.keys()))
    words = sentence.split()

    # Initialize the average word vector, should have the same shape as your word vectors.
    # Use `np.zeros` and pass in the argument of any word's word 2 vec's shape
    avg = np.zeros_like(word_to_vec_map[any_word])
    count = 0

    # Step 2: average the word vectors. You can loop over the words in the list "words".
    for w in words:
        if w in word_to_vec_map:
            avg += word_to_vec_map[w]
            count +=1

    if count > 0:
        avg = avg/count

    return avg

def model(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 400):
    """
    Model to train word vector representations in numpy.
    
    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m,)
    Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations
    
    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """
    
    any_word = next(iter(word_to_vec_map.keys()))
    m = Y.shape[0]                             # number of training examples
    n_y = len(np.unique(Y))                    # number of classes  
    n_h = word_to_vec_map[any_word].shape[0]   # dimensions of the GloVe vectors 
    
    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))
    Y_oh = convert_to_one_hot(Y, C = n_y) 
    for t in range(num_iterations): # Loop over the number of iterations
        cost = 0
        dW = 0
        db = 0
        for i in range(m):          # Loop over the training examples
            avg = sentence_to_avg(X[i], word_to_vec_map)
            z = W @ avg + b
            a = softmax(z)

            cost += -np.dot(Y_oh[i], np.log(a))
            
            # Compute gradients 
            dz = a - Y_oh[i]
            dW += np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db += dz

            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db
            
        assert type(cost) == np.float64, "Incorrect implementation of cost"
        assert cost.shape == (), "Incorrect implementation of cost"
        
        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map) #predict is defined in emo_utils.py

    return pred, W, b

np.random.seed(1)
pred, W, b = model(X_train, Y_train, word_to_vec_map)
```

- Note that the model doesn't get the following sentence correct: "today is not good". The model learns emojis and the appearance of the associated words.

### Unit Test

```python
avg = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vec_map)
print("avg = \n", avg)

def model_test(target):
    # Create a controlled word to vec map
    word_to_vec_map = {'a': [3, 3], 'synonym_of_a': [3, 3], 'a_nw': [2, 4], 'a_s': [3, 2], 'a_n': [3, 4],
                       'c': [-2, 1], 'c_n': [-2, 2],'c_ne': [-1, 2], 'c_e': [-1, 1], 'c_se': [-1, 0],
                       'c_s': [-2, 0], 'c_sw': [-3, 0], 'c_w': [-3, 1], 'c_nw': [-3, 2]
                      }
    # Convert lists to np.arrays
    for key in word_to_vec_map.keys():
        word_to_vec_map[key] = np.array(word_to_vec_map[key])

    # Training set. Sentences composed of a_* words will be of class 0 and sentences composed of c_* words will be of class 1
    X = np.asarray(['a a_s synonym_of_a a_n c_sw', 'a a_s a_n c_sw', 'a_s  a a_n', 'synonym_of_a a a_s a_n c_sw', " a_s a_n",
                    " a a_s a_n c ", " a_n  a c c c_e missing",
                   'c c_nw c_n c c_ne', 'c_e c c_se c_s', 'c_nw c a_s c_e c_e', 'c_e a_nw c_sw', 'c_sw c c_ne c_ne'])

    Y = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    np.random.seed(10)
    pred, W, b = model(X, Y, word_to_vec_map, 0.0025, 110)

    assert W.shape == (2, 2), "W must be of shape 2 x 2"
    assert np.allclose(pred.transpose(), Y), "Model must give a perfect accuracy"
    assert np.allclose(b[0], -1 * b[1]), "b should be symmetric in this example"

    print("\033[92mAll tests passed!")

model_test(model)
```

## LSTM Implementation

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/506dbe78-71d7-48d7-bc2b-2bffb505649e" height="300" alt=""/>
    </figure>
</p>
</div>

### 1. Pad Sequences To The Same Length For Batching

Most deep learning frameworks require that all sequences in the same mini-batch have the same length.

- This is what allows vectorization to work: If you had a 3-word sentence and a 4-word sentence, then the computations needed for them are different (one takes 3 steps of an LSTM, one takes 4 steps) so it's just not possible to do them both at the same time.

* The common solution to handling sequences of **different length** is to use padding. Specifically:
    * Set a maximum sequence length
    * Pad all sequences to have the same length.

### 2. An Embedding Matrix Is a Layer In Keras

The embedding matrix maps word indices to embedding vectors. The embedding vectors are dense vectors of fixed size.

* The embedding matrix can be derived in two ways:
    * Training a model to derive the embeddings from scratch.
    * Using a pretrained embedding.

Dimensions:
    - X: `(batch_size, max_sentence_length)`
    - Output: `(batch_size, max_sentence_length, word_vector )`

```python
def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    
    Arguments:
    X -- array of sentences (strings), of shape (m,)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    m = X.shape[0]                                   # number of training examples
    X_indices = np.zeros((m, max_len))
    for i in range(m):                               # loop over training examples
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            if w in word_to_index:
                X_indices[i, j] = word_to_index[w]
                j = j + 1
    return X_indices
```
### Step 3. pretrained_embedding_layer

1. Initialize the embedding matrix as a numpy array of zeros.The embedding matrix will beL

```
[Word 1 embedding],
[Word 2 embedding],
...
[UNK],
```

- So the embedding matrix is [`number of words + 1, word_embedding_size`]
- Once set, set this layer `trainable=False`, so the embeddings won't be updated during back-prop
word_to_index -> word_to_vec_map

```python
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_size = len(word_to_index) + 1              # adding 1 to fit Keras embedding (requirement)
    any_word = next(iter(word_to_vec_map.keys()))
    emb_dim = word_to_vec_map[any_word].shape[0]    # define dimensionality of your GloVe word vectors (= 50)
      
    emb_matrix = np.zeros((vocab_size, emb_dim))
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] =  word_to_vec_map[word]

    embedding_layer = Embedding(input_dim = vocab_size, output_dim = emb_dim, trainable=False)
    embedding_layer.build((None,)) # Do not modify the "None".  This line of code is complete as-is.
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer
```

-  The embedding layer takes in `[batch, max_length]`, translates each word in the `[max_length]` dimension to one-hot vector (effectively but not likely, for computational cost), multiplies with the embedding matrix, finally outputs `[batch, max_length, embeddings]`

### Step 4 Emojify_V2

- The model takes as input an array of sentences of shape (m, max_len, ) defined by input_shape.
- The model outputs a softmax probability vector of shape (m, C = 5).

```python
def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the Emojify-v2 model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    sentence_indices = Input(input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    embeddings = embedding_layer(sentence_indices) # batch, max_length, embedding_length]
    
    # TODO: why adding a dropout layer? what does the output look like here? 
    
    # Hidden layer is 128
    X = LSTM(units = 128, return_sequences= True)(embeddings) # LSTM Output Shape: (batch_size, max_len, 128)
    X = Dropout(rate = 0.5)(X)  # (batch_size, max_len, 128)
    X = LSTM(units = 128, return_sequences= False)(X)   # Shape: (batch_size, 128), the final hidden state
    X = Dropout(rate = 0.5)(X)
    X = Dense(units = 5)(X)
    X = Activation('softmax')(X)
    
    model = Model(sentence_indices, X)
    
    return model
```

- Dropout **can be added to LSTM**. In this case, Dropout will randomly set the output 0. The purpose is to reduce overfitting.
- The second layer takes in the entire hiddent state from the first layer. E.g., `X = [x₁, x₂, x₃]`, where each x is a hidden state from the first layer. Then:
    1. Timestamp 1: `h1 = LSTM(x1, h0)`. h0 is initialized as zeros
    1. Timestamp 2: `h2 = LSTM(x2, h1)`. 
    1. Timestamp 3: `h3 = LSTM(x3, h3)`. Return h3.

- Keras handles the batch size automatically