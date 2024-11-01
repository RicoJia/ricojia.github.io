---
layout: post
title: Deep Learning - Hands-On Embedding Similarity and Word Emojifier
date: '2022-03-22 13:19'
subtitle: Similarity, Debiasing, Emojifier
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
    - Hands-on
---

TODO: To organize

## Embedding Similarity and Debiasing

embeddings are very computationally expensive to train, most ML practitioners will load a pre-trained set of embeddings.

- GloVe Vectors are 50-vectors
- Cosine similarity:

$$
Cosine sim = uv/|u||v| = cos(theta)
$$

Explain how word embeddings capture relationships between words
Load pre-trained word vectors - from a dictionary;
Measure similarity between word vectors using cosine similarity

Using the GloVe embeddings, if we print the `cos(embedding, e_woman - e_man)`, we get positive results from female names, negative results from male names. Why?

```python
g = word_to_vec_map['woman'] - word_to_vec_map['man']
name_list = ['john', 'marie', 'sophie', 'ronaldo', 'justin', 'justine', 'rahul', 'danielle', 'reza', 'katy', 'yasmin', 'rico', 'suave']
for w in name_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))
```

- This is because `e_woman - e_man` is "the axis of gender". `cos(embedding, e_woman - e_man)` is a "normalized" value of the projection of embedding on `e_woman - e_man`.

- This is useful for debiasing. instead of finding a vector that's orthogonal to the gender axis, we simply subtract the projection component from the vector.

$$
e^{\text{bias_component}} = \frac{e*g}{g * g} * g
e^{\text{debiased}} = e - e^{\text{bias_component}}
$$

- Equalization is applied to pairs of words that you might want to have differ only through the gender property. As a concrete example, suppose that we have neutralized "babysit". "Actress" is closer to "babysit" than "actor." By applying neutralization to "babysit," you can reduce the gender stereotype associated with babysitting. But this still does not guarantee that "actor" and "actress" are equidistant from "babysit." The equalization algorithm takes care of this.

Effectively, we want to keep the analogy between the two words, but only have equal opposite projection along the bias axis. This way, "Actress", "Actor" are equaldistant to "babysit" along the gender axis

![Screenshot from 2024-10-31 16-09-49](https://github.com/user-attachments/assets/bcfe682a-421f-430a-ad78-39038331570c)

In Bolukbasi's work in 2016, we

1. Find the midpoint between the two points of interest
2. Find the corrected position of the midpoint.
3. Find the projection portions of the two points of interest
4. Add `normalized_projection * unit_direction` to the midpoint to get the corrected positions of the two points

```
$$ \mu = \frac{e_{w1} + e_{w2}}{2}\tag{4}$$ 

$$ \mu_{B} = \frac {\mu \cdot \text{bias_axis}}{||\text{bias_axis}||_2^2} *\text{bias_axis}
\tag{5}$$ 

$$\mu_{\perp} = \mu - \mu_{B} \tag{6}$$

$$ e_{w1B} = \frac {e_{w1} \cdot \text{bias_axis}}{||\text{bias_axis}||_2^2} *\text{bias_axis}
\tag{7}$$ 
$$ e_{w2B} = \frac {e_{w2} \cdot \text{bias_axis}}{||\text{bias_axis}||_2^2} *\text{bias_axis}
\tag{8}$$


$$e_{w1B}^{corrected} = \sqrt{{1 - ||\mu_{\perp} ||^2_2}} * \frac{e_{\text{w1B}} - \mu_B} {||e_{w1B} - \mu_B||_2} \tag{9}$$


$$e_{w2B}^{corrected} = \sqrt{{1 - ||\mu_{\perp} ||^2_2}} * \frac{e_{\text{w2B}} - \mu_B} {||e_{w2B} - \mu_B||_2} \tag{10}$$

$$e_1 = e_{w1B}^{corrected} + \mu_{\perp} \tag{11}$$
$$e_2 = e_{w2B}^{corrected} + \mu_{\perp} \tag{12}$$
```

## Word Emojifier

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

### Emojifier-V1

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
