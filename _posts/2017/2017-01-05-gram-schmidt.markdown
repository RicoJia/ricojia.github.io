---
layout: post
title: Math - Gram Schmidt Orthogonolization Process
date: '2017-01-05 13:19'
excerpt: Super useful in finding forming an orthogonal vector basis, e.g., Singular Value Decomposition
comments: true
---

## Background Knowledge

### Glossary

- Inner product: $<a,b> = \vec{a}^T \cdot \vec{b}$, which is a.k.a "dot product"
- Outer product: $a \otimes b = \vec{a} \cdot \vec{b}^T$, which results in a matrix.

### Vector Projection

Projection of a on to b $proj_ba= \frac{<a,b>}{<a,a>}a$

<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/c646e1e5-2e46-4348-a862-be1dc63fd3f0" height="200" alt=""/>
        <figcaption><a href="https://www.google.com/url?sa=i&url=https%3A%2F%2Flearninglab.rmit.edu.au%2Fcontent%2Fv5-projection-vectors.html&psig=AOvVaw3fOwPqOMCslWyRPBmD2gXE&ust=1722182169016000&source=images&cd=vfe&opi=89978449&ved=0CBQQjhxqFwoTCPjUodXKx4cDFQAAAAAdAAAAABAJ">Source: RMIT Learning Lab</a></figcaption>
    </figure>
</p>

## Gram-Schmidt Orthogonalization

**Goal**: given m linearly independent N-dimensional vectors ${v_1 ... v_m}$, transform them into an orthogonal set of vectors.

**Method**: an orthogonal vector is formed by subtracting a vector's projection onto other orthogonal basis, $U$.
1. $u_1 = v_1$
2. $u_2 = v_2 - \frac{<v_2,u_1>}{<u_1,u_1>}u_1$
3. $u_3 = v_3 - \frac{<v_3,u_1>}{<u_1,u_1>}u_1 - \frac{<v_3,u_2>}{<u_2,u_2>}u_2$

**Caveats**

- It's not numerically stable - rounding error in each step can be accumulated. In real life, people use Householder transformation, or Givens rotation.


