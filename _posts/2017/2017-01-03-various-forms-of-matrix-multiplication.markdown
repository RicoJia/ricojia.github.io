---
layout: post
title: Math - Various Useful Forms Of Matrix Multiplication
date: '2017-01-05 13:19'
excerpt: Inner Product, outer Product, and how they are related to matrix multiplication
comments: true
---


## Glossary

- Inner product: $<a,b> = \vec{a}^T \cdot \vec{b}$, which is a.k.a "dot product"
- Outer product: $a \otimes b = \vec{a} \cdot \vec{b}^T$, which results in a matrix.

## Matrix Multiplication And Outer Product

The definition of Matrix Multiplication of $C = AB$ is $C_{ij} = \sum_k A_{ik}B_{kj}$, where A is `mxn`, B is `nxp`

### The matrix product is the sum of the outer product of A's columns and B's rows

That is, $AB = \sum_{k=1}^n a_k b_k^{T}$. Why? Because for any given element $C_{ij}$, we have $C_{ij} = \sum_k A_{ik}B_{kj}$.


