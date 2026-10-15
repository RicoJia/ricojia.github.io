---
layout: post
title: Deep Learning - Flash Attention
date: 2026-10-10 13:19
subtitle: IO-Aware Attention, Tiling, Online Softmax
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Deep Learning
  - Machine-Learning
---
## What FlashAttention Does

FlashAttention computes exact attention without storing the huge $$N\times N$$ attention matrix:

$$
\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Because GPU performance often depends heavily on **memory traffic**. A GPU has two kinds of memory that matter here:

- **HBM** (high-bandwidth memory, the GPU's global memory): large (tens of GB) but relatively slow to read and write.
- **SRAM** (on-chip shared memory): tiny (around 100–200 KB per streaming multiprocessor) but much faster.


1.  **Attention is memory-bound.** The GPU can multiply much faster than it can move the $$N\times N$$ matrix in and out of HBM, so cutting HBM traffic cuts the runtime.
2. **Kernel fusion.** Matmul, mask, softmax, and the multiply by `V` happen in **one** GPU kernel, so the intermediate tiles stay in SRAM instead of making round trips through HBM between kernels.
3. **Memory drops from $$O(N^2)$$ to $$O(N)$$.** Only `Q, K, V`, the output, and two small per-row statistics (`m` and `l`, below) live in HBM. This allows longer sequences or larger batches on the same GPU.

So it reduces expensive HBM reads and writes. Suppose the sequence length is `N`. Then:

$$
Q, K\in\mathbb{R}^{N\times d_k} \quad\Rightarrow\quad QK^T \in \mathbb{R}^{N\times N}
$$


Instead of computing all of $$QK^T$$ at once, it divides `Q, K, V` into small blocks along the sequence dimension:

```
Q:
┌─────┐
│ Q1  │
├─────┤
│ Q2  │
├─────┤
│ Q3  │
└─────┘

K:
┌─────┐
│ K1  │
├─────┤
│ K2  │
├─────┤
│ K3  │
└─────┘
```


> The full $$N\times N$$ attention matrix is never materialized in GPU global memory.

## Refresher - Softmax of a Matrix Normalizes Each Row

In attention, `softmax(A)` is applied **row by row**: each row is one query's scores over all keys, so each row of the result sums to 1:

$$
A= \begin{bmatrix} 1&2&3\\ 2&0&1 \end{bmatrix}, \qquad
\boxed{B_{ij}=\frac{e^{A_{ij}}}{\sum_{k=1}^{3}e^{A_{ik}}}}, \qquad
B=\operatorname{softmax}(A) = \begin{bmatrix} 0.0900&0.2447&0.6652\\ 0.6652&0.0900&0.2447 \end{bmatrix}
$$


## An Example

Take 4×4 matrices Q and K. To get **2×2 attention-score tiles**, split Q and K into **2×4 row blocks**, since (2×4)(4×2) = 2×2. I'll use deliberately simple matrices.

### 1. Define `Q, K, V`

With $$d_k=4$$ (so $$\sqrt{d_k}=2$$), let

$$
Q= \begin{bmatrix} 1&0&0&0\\ 0&1&0&0\\ 0&0&1&0\\ 0&0&0&1 \end{bmatrix}, \qquad
K= \begin{bmatrix} 2&4&0&6\\ 4&2&2&4\\ 6&0&4&2\\ 8&-2&6&0 \end{bmatrix}, \qquad
V = I_4, \qquad
S=\frac{QK^T}{2}.
$$

 just makes the final output very easy to interpret.

### 2. Split into blocks

Split each matrix into **two row blocks**. `V` is split the same way as `K`, since key `j` goes with value `j`:

$$
\begin{gather*}
Q_1= \begin{bmatrix} 1&0&0&0\\ 0&1&0&0 \end{bmatrix}, \quad Q_2= \begin{bmatrix} 0&0&1&0\\ 0&0&0&1 \end{bmatrix}, \qquad
K_1= \begin{bmatrix} 2&4&0&6\\ 4&2&2&4 \end{bmatrix}, \quad K_2= \begin{bmatrix} 6&0&4&2\\ 8&-2&6&0 \end{bmatrix}
\\
V_1= \begin{bmatrix} 1&0&0&0\\ 0&1&0&0 \end{bmatrix}, \quad V_2= \begin{bmatrix} 0&0&1&0\\ 0&0&0&1 \end{bmatrix}
\end{gather*}
$$

### 3. Compute the 2x2 tiles  In Parallel

FlashAttention computes one tile $$S_{ij} = Q_iK_j^T/2$$ at a time. For reference, the full matrix it never materializes is on the right:

$$
S_{11} = \begin{bmatrix} 1&2\\ 2&1 \end{bmatrix}, \quad
S_{12} = \begin{bmatrix} 3&4\\ 0&-1 \end{bmatrix}, \quad
S_{21} = \begin{bmatrix} 0&1\\ 3&2 \end{bmatrix}, \quad
S_{22} = \begin{bmatrix} 2&3\\ 1&0 \end{bmatrix}
\qquad\Longleftrightarrow\qquad
S= \begin{bmatrix} 1&2&3&4\\ 2&1&0&-1\\ 0&1&2&3\\ 3&2&1&0 \end{bmatrix}
$$

Now comes the FlashAttention part.

### 4. `Q_1`, first block `K_1, V_1`

Each **query row** has its **own** running max. In `S_11`, row 1 is $$[1,2]$$ and row 2 is $$[2,1]$$, so both maxima are 2. Exponentiating the shifted scores gives unnormalized softmax weights (not probabilities yet); their row sums are the running denominators:

$$
m= \begin{bmatrix} 2\\ 2 \end{bmatrix}, \qquad
P_{11} = e^{S_{11} - m} = \begin{bmatrix} 0.3679&1\\ 1&0.3679 \end{bmatrix}, \qquad
l= \begin{bmatrix} 1.3679\\ 1.3679 \end{bmatrix}, \qquad
O=P_{11}V_1 = \begin{bmatrix} 0.3679&1&0&0\\ 1&0.3679&0&0 \end{bmatrix}
$$

`Q_2` starts the same way with `S_21`, independently of `Q_1` (see step 7).

### 5. Same `Q_1`, next row block `K_2, V_2`

The new tile is $$S_{12} = \begin{bmatrix} 3&4\\ 0&-1 \end{bmatrix}$$. Update the row-wise max and compute the rescale factor:

$$
m_{\text{tile}} = \begin{bmatrix} 4\\ 0 \end{bmatrix}, \qquad
m_{\text{new}} = \max\left(\begin{bmatrix} 2\\ 2 \end{bmatrix}, \begin{bmatrix} 4\\ 0 \end{bmatrix}\right) = \begin{bmatrix} 4\\ 2 \end{bmatrix}, \qquad
\alpha= \begin{bmatrix} e^{2-4}\\ e^{2-2} \end{bmatrix} = \begin{bmatrix} 0.1353\\ 1 \end{bmatrix}
$$

Row 1's max changed $$2\rightarrow4$$, so row 1 of the old `O` and `l` shrinks by $$0.1353$$; row 2's max stayed 2, so row 2 is unchanged. Next, exponentiate the new tile relative to `m_new` (row 1: $$[3,4]-4=[-1,0]$$; row 2: $$[0,-1]-2=[-2,-3]$$) and multiply by `V_2`:

$$
P_{12} = \begin{bmatrix} 0.3679&1\\ 0.1353&0.0498 \end{bmatrix}, \qquad
P_{12}V_2 = \begin{bmatrix} 0&0&0.3679&1\\ 0&0&0.1353&0.0498 \end{bmatrix}
$$

Add that to the rescaled accumulator $$\alpha O_{\text{old}}$$, and update the denominators with $$l_{\text{new}} = \alpha\, l_{\text{old}} + \operatorname{rowsum}(P_{12})$$:

$$
O_{\text{new}} = \begin{bmatrix} 0.0498&0.1353&0.3679&1\\ 1&0.3679&0.1353&0.0498 \end{bmatrix}, \qquad
l_{\text{new}} = \begin{bmatrix} 0.1353\times1.3679 + 1.3679\\ 1\times1.3679 + 0.1851 \end{bmatrix} = \begin{bmatrix} 1.5530\\ 1.5530 \end{bmatrix}
$$

### 6. Finish `Q_1`: normalize

Divide each row of the accumulator by its denominator:

$$
O_{\text{final},1} = \frac{O_{\text{new}}}{l_{\text{new}}} = \begin{bmatrix} 0.0321&0.0871&0.2369&0.6439\\ 0.6439&0.2369&0.0871&0.0321 \end{bmatrix}
$$

Because $$V=I$$, these rows are the attention probabilities themselves: $$\operatorname{softmax}([1,2,3,4]) = [0.0321,0.0871,0.2369,0.6439]$$ for query 1, and $$\operatorname{softmax}([2,1,0,-1]) = [0.6439,0.2369,0.0871,0.0321]$$ for query 2.

So FlashAttention got exactly the result we'd get from the full matrix, without ever needing to keep $$[1,2,3,4]$$ as a complete score row in global memory.

### 7. Meanwhile `Q_2` can do the same thing

Independently:

```
Q1 worker:

S11 → update state
       ↓
S12 → update state
       ↓
output rows 1,2


Q2 worker:

S21 → update state
       ↓
S22 → update state
       ↓
output rows 3,4
```

For `Q_2`, the tiles `S_21` and `S_22` go through the same updates and give rows 3 and 4. Stacking both workers' outputs gives the complete result:

$$
O_{\text{final},2} = \begin{bmatrix} 0.0321&0.0871&0.2369&0.6439\\ 0.6439&0.2369&0.0871&0.0321 \end{bmatrix}
\quad\Rightarrow\quad
\operatorname{softmax}(S)V = \begin{bmatrix} O_{\text{final},1}\\ O_{\text{final},2} \end{bmatrix} = \begin{bmatrix} 0.0321&0.0871&0.2369&0.6439\\ 0.6439&0.2369&0.0871&0.0321\\ 0.0321&0.0871&0.2369&0.6439\\ 0.6439&0.2369&0.0871&0.0321 \end{bmatrix}
$$

## The Entire FlashAttention Flow

```
Q = 4x4                     K = 4x4
   │                           │
   ▼                           ▼
split rows                  split rows

Q1 = 2x4                   K1 = 2x4
Q2 = 2x4                   K2 = 2x4


       Q1                       Q2
        │                        │
        ▼                        ▼
   Q1 K1^T                 Q2 K1^T
      2x2                     2x2
        │                        │
        ▼                        ▼
 update m,l,O              update m,l,O
        │                        │
        ▼                        ▼
   Q1 K2^T                 Q2 K2^T
      2x2                     2x2
        │                        │
        ▼                        ▼
 update m,l,O              update m,l,O
        │                        │
        ▼                        ▼
 final rows 1,2            final rows 3,4
```

The important pattern is:

$$
\boxed{ Q_iK_1^T \rightarrow \text{update }(m,l,O) \rightarrow Q_iK_2^T \rightarrow \text{update }(m,l,O) \rightarrow O/l }
$$

The two `Q` blocks can be handled in parallel, while the `K/V` blocks are streamed through each `Q` block's running state.
