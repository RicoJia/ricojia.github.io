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



