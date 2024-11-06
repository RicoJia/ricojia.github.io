---
layout: post
title: Deep Learning - Hands-On Embedding Similarity 
date: '2022-03-22 13:19'
subtitle: Similarity and Debiasing
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
    - Hands-on
---

This blog post is a summary of the Coursera Course on Sequence Models

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
e^{\text{bias\_component}} = \frac{e*g}{g * g} * g
e^{\text{debiased}} = e - e^{\text{bias\_component}}
$$

- Equalization is applied to pairs of words that you might want to have differ only through the gender property. As a concrete example, suppose that we have neutralized "babysit". "Actress" is closer to "babysit" than "actor." By applying neutralization to "babysit," you can reduce the gender stereotype associated with babysitting. But this still does not guarantee that "actor" and "actress" are equidistant from "babysit." The equalization algorithm takes care of this.

Effectively, we want to keep the analogy between the two words, but only have equal opposite projection along the bias axis. This way, "Actress", "Actor" are equaldistant to "babysit" along the gender axis

![Screenshot from 2024-10-31 16-09-49](https://github.com/user-attachments/assets/bcfe682a-421f-430a-ad78-39038331570c)

In Bolukbasi's work in 2016, we

1. Find the midpoint between the two points of interest
2. Find the corrected position of the midpoint.
3. Find the projection portions of the two points of interest
4. Add `normalized_projection * unit_direction` to the midpoint to get the corrected positions of the two points

$$ \mu = \frac{e_{w1} + e_{w2}}{2}\tag{4}$$ 

$$ \mu_{B} = \frac {\mu \cdot \text{bias\_axis}}{||\text{bias\_axis}||_2^2} *\text{bias\_axis} \tag{5}$$ 

$$\mu_{\perp} = \mu - \mu_{B} \tag{6}$$

$$ e_{w1B} = \frac {e_{w1} \cdot \text{bias\_axis}}{||\text{bias\_axis}||_2^2} *\text{bias\_axis} \tag{7}$$ 

$$ e_{w2B} = \frac {e_{w2} \cdot \text{bias\_axis}}{||\text{bias\_axis}||_2^2} *\text{bias\_axis} \tag{8}$$

$$e_{w1B}^{corrected} = \sqrt{1 - ||\mu_{\perp} ||^2_2} * \frac{e_{\text{w1B}} - \mu_B} {||e_{w1B} - \mu_B||_2} \tag{9}$$

$$e_{w2B}^{corrected} = \sqrt{1 - ||\mu_{\perp} ||^2_2} * \frac{e_{\text{w2B}} - \mu_B} {||e_{w2B} - \mu_B||_2} \tag{10}$$

$$e_1 = e_{w1B}^{corrected} + \mu_{\perp} \tag{11}$$

$$e_2 = e_{w2B}^{corrected} + \mu_{\perp} \tag{12}$$