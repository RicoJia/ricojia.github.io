---
layout: post
title: Deep Learning - Word Embeddings, Word2Vec
date: '2022-03-18 13:19'
subtitle: Word Representation
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Word Representation

A feature of vocabulary is a vector element that represents an attribute, such as the concept of "fruits", "humans", or more abstract concepts like "dry products", etc. One issue with one-hot vector representation is showing relationship between "features" among words. e.g., in a vocabulary where `man = [0,0,1], orange = [1,0,0], woman = [0,1,0]`, we can't see the "human" concept in `man` and `woman`.

Featurized representation would address this issue. If we design our feature vector to be `[gender, fruit, size]`, then we are able to  use the representation `man = [1,0,0], orange = [0.01,1,0], woman = [-1,0.01,0.01]`. Though ideally, a machine learning model could learn the associations of word sequences in one-hot vectors as well, featurized representation makes the learning process a lot easier.

**The origin of the term "embedding"**: if we visualize our 3-dimensional vocabulary space, we get to "embed" a point for each word.

To visualize high dimensional data and see their relative clustering, one can use [t-SNE](../2017/2017-02-10-tsne.markdown)

Usually, people will use a pretrained network (trained on large corpuses of texts)  and generate embeddings. The embeddings will further go into a learner system

- Facial Recognition is similar, the encoding of faces shold be trained on large amount of data.

### Analogies

In Mikolov et al's work, analogy is formulated as

$$
e_{word1} - e_{word2} \approx e_{word3} - e_{word4}
$$

For example, if the features are: `["gender", "royal", "age"]`,
 the embeddings for words "man" and "woman" are `[1, 0.01, 0]` and `[-1, 0.01, 0]`, king and queen are `[1, 0.97, 0]`, `[-1, 0.95, 0]`, then $e_{man} - e_{woman} \approx e_{king} - e_{queen}$. The below illustration generally describes this relationship

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/f4793284-ce9e-4ef2-9e09-0e6cfbc3ebe8" height="300" alt=""/>
    </figure>
</p>
</div>

To find the analogy "man to woman is like king to ?", we just need to iterate through the entire vocabular, and find the word that roughly corresponds to $e_{king} - e_{man} + e_{woman}$.

One thing to note is that if one wants to do `t-SNE`, the result is after a non-linear mapping, so such a relationship may not hold

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/8553911e-cba0-4abd-8b2a-7773b520b976" height="300" alt=""/>
    </figure>
</p>
</div>

### What Is A Language Model and N-Gram Problem

A language model is essentially a "word predictor". That is, given a sequence of words, what would be the best word to output next? The previous words required for such a prediction is called a "context"

An **n-gram** problem is simply to predict the next word given `n-1` consecutive word. This is to **estimate the probability of occurence of the n-size sequence**, by assuming that the probability of the nth word is only dependent on the previous n-1 words, not the entire sequence.
"Gram" means "grammar", not "Gram Matrix"

$$
\begin{gather*}
P(w_n​∣w_1​,w_2​,…,w_{n−1​})≈P(w_n​∣w_{n−(k−1)​},…,w_{n−1​})
\end{gather*}
$$

- Unigram (1-gram): "I", "love", "NLP"
- Bigram (2-gram): "I love", "love NLP"
- Trigram (3-gram): "I love NLP"

## Embedding Learning

### What is Embedding?

An embedding is basically a linearly-transformed result of an **encoding** so the embedding can be in lower dimension. Say we have a `vocabulary size = 5`, so an input word can be naively represented as an one-hot vector. But one obvious drawback of the one-hot vector is that it is so darn sparse - we want to reduce its dimensionality. In this baby example, now say we want to reduce the representation from 5 to 2, we can define a learnable matrix, where E is `2x5`:

$$
\begin{gather*}
y = Ex
\\
x = [0, 0, 0, 1, 0]^T
\end{gather*}
$$

So this is essentially a linear layer + one-hot vector input. In PyTorch, this is `nn.embedding`. Effectively, it's also a "look up table". Since the input is an one-hot vector, the embedding is one column of E. One guess I have is during backpropagation, only the corresponding column will be updated by the optimizer

### Notes on Embedding Learning

For learning a language model, an consecutive context of N words is nice to have. However, if we just want to learn embeddings, simpler contexts such as last word and skip gram are good enough

## `Word2Vec`

Word2Vec is a shallow neural network model that learns to represent words in a continuous vector space. Developed by researchers at Google in 2013, Word2Vec aims to capture semantic relationships between words based on their co-occurrence patterns in a large corpus of text data. Word2Vec employs two main architectures: Continuous Bag of Words (CBOW) or Skip-gram.

According to the author, CBOW is faster while skip-gram does a better job for infrequent words.

### Skip Gram

Given a "target" word, the Skip-Gram model is to predict its context words. For example, in the "I love to eat pizza", our target words are chosen one-by-one sequentially: "I", "love", ...

if our target word is love, and our window is of size 2, the context words are "[I, to, eat]". Then, we end up with 3 target-context pais: `[love, I], [love, to], [love, eat]`.

Now, we are ready to train: feed "love" into the model, get the output probability distribution, calculate negative log likelihood of `(output, [I, to, eat])`, then backpropagte. Finally, overtime, words that appear together will have similar embeddings.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/7152c159-f002-479e-8180-586e479fb58e" height="300" alt=""/>
        <figcaption><a href="https://arxiv.org/pdf/1301.3781">Source: Mikolov et al. </a></figcaption>
    </figure>
</p>
</div>

There's an [example implementation here](https://www.geeksforgeeks.org/implement-your-own-word2vecskip-gram-model-in-python/#loss-function)

#### Negative Sampling

Note that in the vanilla version of Skip-gram, the softmax $\frac{exp^{y_i}}{\sum_I exp^{y_i}}$ is very expensive to compute, if the vocabulary I is in the order of 10000. Therefore, one can use:

- Hierarchical Softmax Classifier
- Or negative sampling

The key idea of negative sampling is to transform the multi-classification problem into a series of binary classification problems. How?

Consider the sentence: "The cat sat on the mat." Say the target word is "cat". We can construct the "positive" set with

```
("cat", "the")
("cat", "sat")
```

We also randomly selected words like "computer", "blue", "run" not related to "cat", and construct a negative set

```
("cat", "blue")
("cat", "run")
```

So, the output layer could be a single neuron and we have a binary-classification problem.

### Continuous Bag of Words (CBoW)

CBoW was introduced by Mikolov et al. in 2013. Its purpose is to learn the embeddings, and a neural network that predicts the next word given `n` consecutive words (n-gram problem). [Here, is an excerpt of its minimum implementation](https://gist.github.com/GavinXing/9954ea846072e115bb07d9758892382c):

```python
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

class CBOW(nn.Module):

    def __init__(self, context_size=2, embedding_size=100, vocab_size=None):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(embedding_size, vocab_size)

    def forward(self, inputs):
        lookup_embeds = self.embeddings(inputs)
        embeds = lookup_embeds.sum(dim=0)
        out = self.linear1(embeds)
        out = F.log_softmax(out)
        return out
```

- In Bag of Words, word order does NOT matter. So if you wonder about why these `embeds` is the sum of `lookup_embeds`, that's why.

## `Neural Probabilistic Language Model`

- In some scenarios, like "Neural Probabilistic Language Model" [2], word order does matter. [Here is an implementation by Younes Dahami](https://medium.com/@dahami/a-neural-probabilistic-language-model-breaking-down-bengios-approach-4bf793a84426)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralProbabilisticLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim = 80):
        super().__init__()
        # Word embeddings (Embedding matrix)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.tanh = nn.Tanh()  
        self.linear2 = nn.Linear(hidden_dim, vocab_size)  

    def forward(self, inputs):

        # Flatten embeddings
        embeds = self.embeddings(inputs).view((1, -1))  
        out = self.linear1(embeds) 
        out = self.tanh(out)  
        out = self.linear2(out)  
        # Log-softmax for probability distribution
        log_probs = torch.log_softmax(out, dim=1)  
        return log_probs

vocab_size = 100
embedding_dim = 50
context_size = 3
hidden_dim = 128


model = NeuralProbabilisticLanguageModel(vocab_size = vocab_size, embedding_dim=embedding_dim, context_size=context_size)
# Negative log-likelihood loss
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)  

# Example training data (context and target word)
context = torch.tensor([2, 45, 67], dtype=torch.long)  # Example context
target = torch.tensor([85], dtype=torch.long)  # Example target word

# Training loop (for one step)
model.zero_grad()  # Zero the gradients
log_probs = model(context)  # Forward pass
loss = loss_function(log_probs, target)  # Compute the loss
loss.backward()  # Backpropagation
optimizer.step()  # Update the model parameters
```

## Global Vectors for Word Representation (GloVe)

The goal of GloVe is to directly learn two sets of word embeddings, such that two words from one embedding set each's similarity (their dot product) approximates the log of their co-occurence times.

$$
\begin{gather*}
minimize \sum_{I=10000} \sum_{J = 10000} f(x_{ij})(\theta_i^T e_j - log(x_{ij}) + b_i + b_j)^2
\end{gather*}
$$

- $x_{ij}$: the number of times that words `i, j` occur together.
- $\theta_i$, $e_j$ are two independently trained sets word embeddings of words `i, j`
- $f(x_{ij})$ is a weighting term of, such that when `i`, `j` never co-occur, $log(0) = -inf$, we are numerically stable.
  - A uniform probability for each word may not reflect the true weight of the word pair
  - But if we just consider the frequencies of the two words, they are a bit too harsh.

Some code to illustrate this purpose:

```python

def initialize_parameters(vocab_size, vector_size):
    W = np.random.uniform(-0.5, 0.5, (vocab_size, vector_size))
    W_tilde = np.random.uniform(-0.5, 0.5, (vocab_size, vector_size))
    biases = np.zeros(vocab_size)
    biases_tilde = np.zeros(vocab_size)
    return W, W_tilde, biases, biases_tilde

def build_cooccurrence_matrix(tokenized_corpus, word_to_id, window_size=2):
    vocab_size = len(word_to_id)
    cooccurrence_matrix = defaultdict(Counter)
    for tokens in tokenized_corpus:
        token_ids = [word_to_id[word] for word in tokens]
        for center_i, center_id in enumerate(token_ids):
            context_indices = list(range(max(0, center_i - window_size), center_i)) + \
                              list(range(center_i + 1, min(len(token_ids), center_i + window_size + 1)))
            for context_i in context_indices:
                context_id = token_ids[context_i]
                distance = abs(center_i - context_i)
                increment = 1.0 / distance  # Weighting by inverse distance
                cooccurrence_matrix[center_id][context_id] += increment
    return cooccurrence_matrix

def train_glove(cooccurrence_matrix, word_to_id, vector_size=50, iterations=100, learning_rate=0.05):
    vocab_size = len(word_to_id)
    W, W_tilde, biases, biases_tilde = initialize_parameters(vocab_size, vector_size)
    global_cost = []
    for it in range(iterations):
        total_cost = 0
        for i, j_counter in cooccurrence_matrix.items():
            for j, X_ij in j_counter.items():
                weight = weighting_function(X_ij)
                w_i = W[i]
                w_j = W_tilde[j]
                b_i = biases[i]
                b_j = biases_tilde[j]
                inner_cost = np.dot(w_i, w_j) + b_i + b_j - np.log(X_ij)
                cost = weight * (inner_cost ** 2)
                total_cost += 0.5 * cost

                # Compute gradients
                grad_main = weight * inner_cost * w_j
                grad_context = weight * inner_cost * w_i
                grad_bias_main = weight * inner_cost
                grad_bias_context = weight * inner_cost

                # Update parameters
                W[i] -= learning_rate * grad_main
                W_tilde[j] -= learning_rate * grad_context
                biases[i] -= learning_rate * grad_bias_main
                biases_tilde[j] -= learning_rate * grad_bias_context
        global_cost.append(total_cost)
        print(f"Iteration {it+1}, Cost: {total_cost}")
    return W + W_tilde  # Combine word and context word vectors
```

## Refereces

[1] [Mikolov, T., Yih, W., & Zweig, G. (2013). Linguistic Regularities in Continuous Space Word Representations. In Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), pages 746-751, Atlanta, Georgia. Association for Computational Linguistics.](https://aclanthology.org/N13-1090.pdf)

[2] [Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A Neural Probabilistic Language Model. Journal of Machine Learning Research, 3, 1137-1155](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
