---
layout: post
title: Deep Learning - Word Embeddings
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



## Refereces

[1] [Mikolov, T., Yih, W., & Zweig, G. (2013). Linguistic Regularities in Continuous Space Word Representations. In Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), pages 746-751, Atlanta, Georgia. Association for Computational Linguistics.](https://aclanthology.org/N13-1090.pdf) 
