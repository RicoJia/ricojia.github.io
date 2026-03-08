---
layout: post
title: Entropy Bottleneck
date: 2026-02-08 13:19
subtitle: Entropy Encoding, Range Encoding
comments: true
header-img: img/post-bg-kuaidi.jpg
tags:
  - CUDA
---


## Problem Setup

Suppose one encoder output (latent vector) is `z = [0.13, -1.82, 0.07]`. Storing these as float32 costs $3 \times 32 = 96$ bits. Entropy coding requires **discrete symbols**, so we first quantize by rounding:

```
z     = [0.13, -1.82,  0.07]
z_hat = [0,    -2,     0   ]
```

After processing many inputs, the empirical symbol distribution might look like:

| Symbol | Probability |
|--------|-------------|
| 0      | 0.60        |
| -1     | 0.20        |
| -2     | 0.10        |
| 1      | 0.10        |

The **Shannon entropy** gives the theoretical minimum bits per symbol on average:

$$H = -\sum_x p(x)\log_2 p(x)$$

$$H = -0.6\log_2(0.6) - 0.2\log_2(0.2) - 0.1\log_2(0.1) - 0.1\log_2(0.1) \approx 1.57 \text{ bits}$$

So each quantized symbol needs only ~1.57 bits on average — far fewer than 32 bits for float32.

**Entropy coding** (e.g. Huffman or arithmetic coding) assigns shorter bit-strings to more frequent symbols:

| Symbol | Code |
|--------|------|
| 0      | `0`    |
| -1     | `10`   |
| -2     | `110`  |
| 1      | `111`  |

The sequence `[0, -2, 0]` encodes to `0 110 0` = **5 bits** instead of 96.

## Neural Network As Encoder

The encoder + entropy bottleneck is trained end-to-end to minimise:

$$\mathcal{L} = D + \lambda R, \qquad R = -\sum \log_2 p(\hat{z})$$

where $D$ is reconstruction distortion and $R$ is the estimated bit-rate. Minimising $R$ drives the network to:

- **Concentrate latents in high-probability regions** — $p$ assigns high probability to the values the encoder produces, making them cheap to code.
- **Cluster values near zero** — if $p$ is peaked at zero, pushing activations toward zero minimises bit cost.
- **Use a peaked (low-entropy) distribution** — fewer distinct symbols → fewer bits per symbol on average.

### During Training: Quantization with Noise

Hard rounding $\hat{y} = \text{round}(y)$ has zero gradient everywhere, so it cannot be used directly for backprop. Instead, uniform noise is added:

$$\tilde{y} = y + u, \qquad u \sim \mathcal{U}(-0.5,\, 0.5)$$

This **approximates quantization** in expectation while keeping gradients flowing. $\tilde{y}$ is **not** an integer — it is a continuous value that statistically behaves like a rounded one. The rate loss is computed using $p(\tilde{y})$ as a proxy for $p(\hat{y})$.

**But noise is random — how is it differentiable?** For a fixed sampled $u$, $\tilde{y} = y + u$ is just a shift, so:

$$\frac{\partial \tilde{y}}{\partial y} = 1$$

Backprop flows through as identity. The randomness is in $u$, but the mapping is smooth in $y$. Training then optimises the expected loss over noise:

$$\mathbb{E}_{u \sim \mathcal{U}(-0.5,\,0.5)}\bigl[\mathcal{L}(y + u)\bigr]$$

One sampled $u$ per forward pass is a Monte Carlo estimate of this expectation — the same idea as dropout. Gradients are stochastic but unbiased for the true objective.

### Entropy Bottleneck — Learning the CDF

Rather than storing a fixed probability table, the entropy bottleneck learns a flexible CDF $F(y)$ via a small neural network (`_logit_cumulative()`), which implements a learned monotonic function $g(x)$ and outputs:

$$F(x) = \sigma(g(x))$$

Using a learned $g$ instead of plain $\sigma(x)$ lets the distribution be skewed, heavy-tailed, or asymmetric — whatever best fits the latent statistics.

The probability mass assigned to quantized integer $k$ is:

$$P(k) = F(k + 0.5) - F(k - 0.5)$$

**Why model the CDF instead of the PDF?**  
Because the probability of landing in bin $k$ after rounding is exactly:

$$P(k) = \int_{k-0.5}^{k+0.5} p(x)\,dx = F(k+0.5) - F(k-0.5)$$

Computing this via CDF differences is numerically stable and fully differentiable.

**Example** (toy CDF = sigmoid $\sigma(x)$):

| Integer $k$ | $F(k+0.5)$ | $F(k-0.5)$ | $P(k)$ | Bits $= -\log_2 P(k)$ |
|:-----------:|:----------:|:----------:|:------:|:---------------------:|
| 0           | $\sigma(0.5)=0.622$ | $\sigma(-0.5)=0.378$ | 0.244 | 2.03 |
| 2           | $\sigma(2.5)=0.924$ | $\sigma(1.5)=0.818$  | 0.106 | 3.24 |

**In summary**, `EntropyBottleneck` is a learnable histogram estimator. The full pipeline is:

$$\text{continuous latent} \;\to\; \hat{y} \;\to\; P(k) = F(k{+}0.5)-F(k{-}0.5) \;\to\; \text{bits} = -\log_2 P(k) \;\xrightarrow{\text{inference only}}\; \text{bitstream}$$

**Training** — real entropy coding is **not** used. The model computes the expected bit cost directly:

```python
bits = -torch.log2(likelihood)   # likelihood = P(k) from the learned CDF
```

Why not run a real coder during training? Arithmetic/range coding is non-differentiable, slow, and unnecessary — the expected rate $-\log_2 P(k)$ is mathematically equivalent to the true bit cost in the limit.

**Inference** — real entropy coding **is** used. CompressAI's `EntropyBottleneck` uses **range coding** (a practical variant of arithmetic coding) via `compress()` / `decompress()`.

### Usage in D-PCC

```
latent_feats (floats)
       │
  feats_eblock(latent_feats)
       │
       ├─ latent_feats_hat          ← quantized floats, passed to decoder (what you transmit)
       └─ latent_feats_likelihoods  ← per-element P(k), used only for the bpp loss
```

```python
latent_feats_hat, latent_feats_likelihoods = self.feats_eblock(latent_feats)
feats_size = torch.log(latent_feats_likelihoods).sum() / (-math.log(2))
feats_bpp  = feats_size / points_num
```

- `latent_feats_hat` — still floating-point values; the decoder consumes them as-is. **This is what you would transmit.**
- `latent_feats_likelihoods` — probabilities under the learned $F$, used to compute `feats_bpp` $= -\sum \log_2 p / N$ as a surrogate loss. No actual bits are produced here.
- `feats_bpp` tells you the **theoretical minimum** bits-per-point achievable if you applied range coding. The model's `forward()` never calls the range coder.

**Should you call `EntropyBottleneck.compress()` over your communication channel?**

Yes — if your channel accepts a byte payload, calling `compress()` on `latent_feats_hat` will range-code those quantized values using the learned $F$ and can reduce transmission size to approximately `feats_bpp` bits per point, which is close to the Shannon optimum. If your channel already transmits raw float32 tensors (e.g. a socket sending numpy arrays), the range coder adds no benefit unless you need to hit a strict bandwidth budget. The short answer: **use `compress()` / `decompress()` whenever you need an actual bitstream**; skip it if raw tensors are acceptable.

---

## Range Encoding

Range coding is a practical arithmetic coder. It encodes a sequence of symbols into a single integer whose bit-length approaches the Shannon entropy.

**Step 1 — Assign probability ranges**

| Symbol | $P$ | Cumulative range |
|--------|-----|-----------------|
| A | 0.50 | $[0,\ 50)$ |
| B | 0.30 | $[50,\ 80)$ |
| C | 0.20 | $[80,\ 100)$ |

**Step 2 — Initialise the interval**

$$\text{low} = 0,\quad \text{high} = 9999,\quad \text{range} = 10000$$

**Step 3 — Encode each symbol** (example sequence: `[A, C]`)

*Encode A* ($[0, 50)$):

$$\text{low}' = 0 + 10000 \times \tfrac{0}{100} = 0 \qquad \text{high}' = 0 + 10000 \times \tfrac{50}{100} - 1 = 4999$$

*Encode C* ($[80, 100)$), range $= 5000$:

$$\text{low}'' = 0 + 5000 \times \tfrac{80}{100} = 4000 \qquad \text{high}'' = 0 + 5000 \times \tfrac{100}{100} - 1 = 4999$$

**Step 4 — Emit a number in $[4000, 4999]$**, e.g. $4500 = 1000110010100_2$. That binary string is the compressed bitstream.

**$4500$ takes 13 bits — does efficiency improve with longer sequences?**

Yes, dramatically. The 13-bit cost is dominated by the fixed overhead of expressing a number within the initial range $[0, 9999]$ ($\approx \log_2 10000 = 13.3$ bits total), regardless of how many symbols were encoded. With only 2 symbols that is $13/2 = 6.5$ bits per symbol — far above the 3.32-bit Shannon limit. With $n$ symbols the total bitstream length approaches:

$$\underbrace{\log_2(\text{initial range})}_{\text{fixed overhead}} + \sum_{i=1}^{n} -\log_2 P(x_i)$$

As $n$ grows the overhead is amortised and the per-symbol cost converges to the entropy. In practice, range coders periodically **renormalise** — rescaling the interval and flushing bits into a fixed-width register — allowing arbitrarily long sequences in bounded memory.

**Why does this achieve compression?**

The final interval width is $\propto P(A) \times P(C) = 0.5 \times 0.2 = 0.1$, so the interval spans $10000 \times 0.1 = 1000$ values. The bits needed to identify one value in that sub-range:

$$\log_2\!\frac{10000}{1000} = -\log_2(0.5) - \log_2(0.2) = 1 + 2.32 = 3.32 \text{ bits}$$

Exactly the Shannon limit $\sum -\log_2 P(x_i)$.

**Decoding** is the reverse: given $4500$, check which symbol range it falls in under the current scaled interval, peel off that symbol, then narrow the interval and repeat.

### Connection to EntropyBottleneck

`EntropyBottleneck` provides $P(k) = F(k{+}0.5) - F(k{-}0.5)$ for each quantized value $k$. The range coder then encodes the sequence $[k_1, k_2, \ldots]$ into a bitstream of length $\approx \sum -\log_2 P(k_i)$ — matching the `feats_bpp` quantity computed during training.

**What does `compress()` actually take as input, and what does it return?**

- **Input**: `latent_feats_hat` of shape `(B, C, N)`. These are **integer-valued floats** (e.g. `0.0, -2.0, 1.0`) — `compress_ai.EntropyModel.compress()` already rounds them in its `quantize()` function. Therefore values like `1.4` do not appear.
- **Output**: a list of byte strings (one per batch item), plus the tensor shape as side information needed by the decoder.

**Does the decoder need an EOS symbol to know when to stop?**

No. In CompressAI-style neural compression, the latent shape `(B, C, N)` is transmitted as a small header alongside the bitstream. The decoder instantiates a tensor of that exact shape and calls `decompress(strings, shape)` — it knows exactly how many symbols to decode. No EOS symbol is required, and point cloud size does **not** need to be fixed; the shape is just sent as side information.

### Effect of World-Frame Position on Training

Training the encoder on point clouds at arbitrary world positions (e.g. a LiDAR scan centred at $(1000, 500, 0)$ vs. $(−200, 800, 0)$) **hurts compression quality**. The encoder sees wildly varying absolute coordinates, so the latent distribution is non-stationary — different inputs produce very different $\hat{z}$ values, making it hard for the entropy bottleneck to learn a peaked, low-entropy distribution.

**Standard fix**: normalise each point cloud before the encoder — subtract the centroid (and optionally scale to unit extent). This makes the input distribution stationary across scenes and lets the encoder + bottleneck learn a compact, consistent latent space. If your system must preserve absolute world coordinates, transmit the centroid separately and add it back after decoding.
