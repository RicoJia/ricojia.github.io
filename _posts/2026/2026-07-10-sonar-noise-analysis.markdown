---
layout: post
title: "[Robotics] Sonar Noise Analysis"
date: 2026-07-10 13:19
subtitle: phasor, rayleigh distribution, exponential distribution, gamma distribution, TVG simulation
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - robotics
---
Sonar noise have two independent characteristics:

1. **Intensity noise:** usually multiplicative Gamma speckle.
2. **Range noise:** usually additive, spatially correlated residuals.
3. Time varying gain : the receiver increases gain for later-arriving echoes because later arrival corresponds to greater range.

Below, section 1 - 7 details the rationale behind applying gamma speckle multiplicatively to individual beams. Section 8 is how to introduce spatial correlation between beams using a Gaussian Kernel. Section 9 is about how to model TVG (time-varying gain) at the receiver end

## 1 - Phasor

Euler’s formula of a single complex number

$$  
e^{jx}=\cos x+j\sin x.  
$$

A sinusoidal echo can therefore be represented by the complex number

$$  
z=X+jY=Ae^{j\phi}=A(\cos\phi+j\sin\phi),  
$$

where

$$  
A=\sqrt{X^2+Y^2},  
\qquad  
\phi=\operatorname{atan2}(Y,X).  
$$

For example, with (X=3) and (Y=4),

$$  
z=3+4j,  
\qquad  
A=\sqrt{3^2+4^2}=5,  
\qquad  
\phi=\operatorname{atan2}(4,3)\approx53.1^\circ.  
$$

The physical waveform is the real part of the rotating complex signal:, whose frequency is fixed $\omega$:

$$  
s(t)=\operatorname{Re}[(3+4j)e^{j\omega t}].  
$$

Expanding the above:
 $$  
(3+4j)e^{j\omega t} =

(3+4j)\bigl(\cos(\omega t)+j\sin(\omega t)\bigr)  

=3\cos(\omega t)-4\sin(\omega t)  
=5\cos(\omega t+53.1^\circ).  

$$

**So any sinusoid can be represented as the real part of a complex phasor, a compact representation.**

$$  
s(t)=A\cos(\omega t+\phi).  
$$

## 2- Rayleigh Distribution

A Rayleigh distribution describes the **magnitude of a 2D vector whose two components are independent zero-mean Gaussian random variables**. A sonar echo can be represented as a phasor:

$$  
z = X + jY  
$$

**Assume X and Y have the same standard deviation (important assumption)**:

$$  
X \sim \mathcal N(0,\sigma^2),  
\qquad  
Y \sim \mathcal N(0,\sigma^2)  
$$

The measured echo envelope is the magnitude is:

$$  
A = |z| = \sqrt{X^2+Y^2}  
$$

That magnitude (A) follows **a Rayleigh distribution.** Suppose one sonar resolution cell gives these in-phase and quadrature values:

| Sample |    (X) |    (Y) |    (A) |
| -----: | -----: | -----: | -----: |
|      1 |  (0.6) |  (0.8) | (1.00) |
|      2 | (-0.3) |  (0.4) | (0.50) |
|      3 |  (1.0) | (-0.2) | (1.02) |
|      4 | (-0.7) | (-0.7) | (0.99) |
|      5 |  (0.1) |  (0.2) | (0.22) |

Notice that  Gaussian components can be positive or negative, but **the magnitude cannot be negative**:

$$  
A \ge 0  
$$

A follows the Rayleigh probability density:

 $$  
f_A(a) =

\frac{a}{\sigma^2}  
\exp\left(-\frac{a^2}{2\sigma^2}\right),  
\qquad a\ge0  
$$

where $\sigma$ is the standard deviation of X and Y.  with \(X\) and \(Y\) independent. Then the joint distribution of (X,Y) is (w.r.t each realization (x,y)):

$$
f_{X,Y}(x,y)
=
\frac{1}{2\pi\sigma^2}
\exp\left(
-\frac{x^2+y^2}{2\sigma^2}
\right).
$$

Using

$$
x=r\cos\theta,
\qquad
y=r\sin\theta,
\qquad
x^2+y^2=r^2,
$$

we get

$$
f_{X,Y}(r\cos\theta,r\sin\theta)
=
\frac{1}{2\pi\sigma^2}
\exp\left(
-\frac{r^2}{2\sigma^2}
\right).
$$

For ($\sigma=1$):

$$  
f_A(a)=r\exp\left(-\frac{r^2}{2}\right)  
$$

![](https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/Rayleigh_distributionPDF.svg/1920px-Rayleigh_distributionPDF.svg.png)

### Mean and standard deviation

For a Rayleigh random variable,

$$  
E[A] =

\sigma\sqrt{\frac{\pi}{2}}  
$$

and:

$$  
\operatorname{std}(A) =

\sigma\sqrt{\frac{4-\pi}{2}}  
$$

For ($\sigma=1$):

$$  
E[A]\approx1.253  
$$

$$  
\operatorname{std}(A)\approx0.655  
$$

Therefore, coefficient of variation (CV) doesn't depend on $\sigma$:

$$  
CV =

\frac{\operatorname{std}(A)}{E[A]}  = \frac{0.655}{1.253}  
\approx0.523  

$$

---

## 3 - Sonar Pings Are Phasors

Inside one sonar resolution cell, the return may contain echoes from many small scatterers: sand grains, small rocks, etc. Each scatterer returns a complex acoustic wave with a different phase. Imagine many tiny scatterers inside one sonar beam:

```text
sand grain 1 → small echo with phase 20°
sand grain 2 → small echo with phase 170°
sand grain 3 → small echo with phase 290°
...
```

Their complex echoes add:

$$  
z=\sum_k a_k e^{j\phi_k} = a_1e^{j\phi_1}  
+  
a_2e^{j\phi_2}  
+  
\cdots  
+  
a_ne^{j\phi_n}  
$$

When there are many scatterers with roughly random phases, the real and imaginary parts of (z) become approximately Gaussian. The real and imaginary parts of the sum become approximately Gaussian:

$$  
X\sim\mathcal N(0,\sigma^2),  
\qquad  
Y\sim\mathcal N(0,\sigma^2)  
$$
For example, if a Rayleigh-distributed amplitude has $CV \approx 0.523$,

 $$  
CV_A =

\frac{\sigma_A}{\mu_A}

\sqrt{\frac{4-\pi}{\pi}}  
\approx 0.523  
$$

It means: If the logged sonar value is the raw linear echo amplitude from one independent look at a rough surface, its standard deviation should be about (52.3%) of its mean.

---

## 4 - Power Density Follows The Gamma Distribution

Define intensity or power as

$$  
i=r^2.  
$$
For a transformation of random variables,
$$  
f_I(i) =

f_R\left(\sqrt{i}\right)  
\left|  
\frac{d\sqrt{i}}{di}  
\right|.  
$$
Where the derivative is:

 $$  
\frac{d\sqrt{i}}{di} =

\frac{1}{2\sqrt{i}}.  
$$
Substitute the Rayleigh PDF:

$$  
f_I(i) =

\frac{\sqrt{i}}{\sigma^2}  
\exp\left(-\frac{i}{2\sigma^2}\right)  
\frac{1}{2\sqrt{i}}

=

\frac{1}{2\sigma^2}  
\exp\left(-\frac{i}{2\sigma^2}\right),  
\qquad i\ge0.  

$$

This is an exponential distribution with mean

$$  
E[I]=2\sigma^2.  
$$

Therefore,

$$  
\boxed{  
R\sim\operatorname{Rayleigh}(\sigma)  
\quad\Longrightarrow\quad  
I=R^2\sim\operatorname{Exponential}\left(\text{mean}=2\sigma^2\right).  
}  
$$

### 4 - 1 Gamma Distribution

On the other hand, A Gamma-distributed random variable (G) with shape (k) and scale ($\theta$) has PDF

$$  
\boxed{  
f_G(g) =

\frac{1}{\Gamma(k)\theta^k}  
g^{k-1}  
\exp\left(-\frac{g}{\theta}\right),  
\qquad g\ge0.  
}  
$$
Here:

- (k>0) is the **shape** parameter;
- ($\theta$>0) is the **scale** parameter;
- ($\Gamma(k)$) is the Gamma function.

For positive integers,

$$  
\Gamma(k)=(k-1)!.  
$$
The mean and variance are

$$  
E[G]=k\theta,  
$$
$$  
\operatorname{Var}(G)=k\theta^2.  
$$
The coefficient of variation is

$$  
CV = \frac{\sqrt{\operatorname{Var}(G)}}{E[G]}

= \frac{1}{\sqrt{k}}.  
$$

### 4 - 2 The exponential distribution is Gamma with k=1

Set

$$  
k=1.  
$$

The Gamma PDF becomes
$$  
f_G(g) =

\frac{1}{\Gamma(1)\theta}  
g^0  
\exp\left(-\frac{g}{\theta}\right).  
$$

Since

$$  
\Gamma(1)=1,
g^0=1
$$

we obtain $$  
f_G(g)
=

\frac{1}{\theta}  
\exp\left(-\frac{g}{\theta}\right).  
$$

That is exactly the exponential PDF.

---

## 5 - Average of multiple pings produce a Gamma distribution

Suppose the sonar takes (L) independent measurements of the same underlying return:

$$  
I_1,I_2,\ldots,I_L.  
$$

Each single-look intensity is exponential:

$$  
I_\ell  
\sim  
\operatorname{Gamma}(1,\theta).  
$$
Their sum is
$$  
T =

\sum_{\ell=1}^{L}I_\ell.  
$$

The sum of (L) independent Gamma variables with the same scale is

$$  
\boxed{  
T  
\sim  
\operatorname{Gamma}(L,\theta).  
}  
$$
Its PDF is $$  
f_T(t) =

\frac{1}{\Gamma(L)\theta^L}  
t^{L-1}  
\exp\left(-\frac{t}{\theta}\right).  
$$
So the chain is

$$  
(I_1+\cdots+I_L)  
\sim  
\operatorname{Gamma}(L,\theta).  
$$
Usually the sonar averages the measurements:
 $$  
\bar I =

\frac{1}{L}  
\sum_{\ell=1}^{L}I_\ell.  
$$
If

$$  
I_\ell  
\sim  
\operatorname{Gamma}(1,\theta),  
$$
then

$$  
\boxed{  
\bar I  
\sim  
\operatorname{Gamma}  
\left(  
L,\frac{\theta}{L}  
\right).  
}  
$$
Its variance becomes

$$  
\operatorname{Var}(\bar I) = L\left(\frac{\theta}{L}\right)^2

= \frac{\theta^2}{L}.  
$$Therefore, average more measurements reduces the relative fluctuation.

$$  
CV_{\bar I} = \frac{\theta/\sqrt L}{\theta}

= \boxed{\frac{1}{\sqrt L}}.  
$$

At pixel location ((r,c)), let the underlying clean intensity be

$$  
I_{\text{clean}}[r,c]=\theta[r,c].  
$$

A single-look measured intensity is modeled as

$$  
I_\ell[r,c]  
\sim  
\operatorname{Gamma}  
\left(  
1,\theta[r,c]  
\right),  
$$
where the second parameter is the **scale**. Because shape (=1), this is an exponential distribution with

$$  
E[I_\ell[r,c]] =

\theta[r,c] =

I_{\text{clean}}[r,c].  
$$

Now  we want to look at the normalized intensity distribution:

$$  
S[r,c] =

\frac{\bar I[r,c]}  
{I_{\text{clean}}[r,c]}.  
$$

Then it's

$$  

S[r,c]  
\sim  
\operatorname{Gamma}  
\left(  
L,  
\frac{1}{I_{\text{clean}}[r,c]}  
\frac{I_{\text{clean}}[r,c]}{L}  
\right)
=

\operatorname{Gamma}  
\left(  
L,\frac{1}{L}  
\right)  
.
$$

## 6 - Example with Hypothetical Numbers

As we increase the number of measurements of the same seabed surface:

```text
L = 1   Exponential: strongly skewed, very noisy
L = 2   Still skewed, but smoother
L = 4   More concentrated near 1
L = 8   Considerably smoother
L → ∞   Concentrates around 1
```

|Looks (L)|Speckle CV|
|--:|--:|
|1|1.000|
|2|0.707|
|4|0.500|
|5|0.447|
|6|0.408|
|10|0.316|

Therefore, if we measured $CV \in [0.41, 0.49]$  corresponds approximately to

$$  
L=\frac{1}{CV^2}  \approx4.2\text{–}5.9.  

$$

## 7 - Apply Gamma Distributed Noise Multiplicatively On the Same Beam

Gaussian additive noise would say $$  
I_{\text{measured}} =

I_{\text{clean}}+\epsilon.  
$$
That implies the same absolute noise strength for dark and bright returns and can produce negative intensities.

Gamma speckle says:
$$  
I_{\text{measured}} =

I_{\text{clean}}S.  
$$

Where S is the Gamma noise. This better represents speckle because:

- intensity stays positive;
- bright returns fluctuate more in absolute magnitude;
- the relative variability is approximately constant;
- averaging multiple exponential looks naturally produces Gamma statistics.

The simplest model is independent Gamma speckle:

```python
speckle = np.random.gamma(
    shape=L,
    scale=1.0 / L,
    size=clean_intensity.shape,
)

observed_intensity = clean_intensity * speckle
```

---

## 8 - Use Gaussian Kernel To Add Spatial Correlation To Noise

Suppose you start with white noise independently at every cell.:

$$  
w[r,c]\sim\mathcal N(0,1)  
$$
Then filter it:

$$  
g = K*w,  
$$

where (K) is a Gaussian kernel.

After convolution, adjacent cells share many of the same input random values:

```text
output cell A uses nearby white-noise cells
output cell B uses almost the same nearby cells
```

Therefore, `(g[r,c])` and `(g[r,c+1])` become correlated.

Of course, an anisotropic kernel such as

```python
sigma=(0.82, 0.75)
```

can create different correlation strengths in the two grid directions.

Then we add directly additive range noise:

$$  
R_{\text{observed}} =

R_{\text{clean}}+g.  
$$
One characteristic here is sonar signals are "heavy-tailed".  A regular gaussian noise is: `0.03, -0.04, 0.02, 0.06, -0.05`, but a heavy tailed one is  `0.03, -0.04, 0.02, 2.50, 0.06.`  "Tail" refers to the shape of the probability distribution, instead of its physical location.

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQdTyHBQ2qFzR9gK1ncfSsDQN5Dg0oOcamXWu1koT3a7lHiMZznljpNJSs&s=10)

In code, to model a gaussian noise with heavy tail

```python
base = gaussian_blur(
    np.random.randn(rows, cols),
    sigma=(0.82, 0.75),
)

amplitude = np.exp(
    beta * gaussian_blur(
        np.random.randn(rows, cols),
        sigma=(4.0, 4.0),
    )
)

noise = base * amplitude
```

---

## 9 - TVG Modelling

TVG means **time-varying gain**. For synthetic data, TVG variation should usually be modeled as a **smooth range-dependent multiplier**, not independent random noise per pixel.

$$  
I_{\mathrm{output}}(r) =

I_{\mathrm{raw}}(r)G_{\mathrm{TVG}}(r).  
$$

Substituting gives
$$  
I_{\mathrm{synthetic}}[r,c] =

I_{\mathrm{clean}}[r,c]  
\left(\frac{r_0}{R[r,c]}\right)^p,  
$$
where (p) is a residual exponent. For example:

$$  
p=1.2,  
\qquad  
r_0=10\text{ m}.  
$$
Then:

- at (10) m, multiplier (=1);
- at (20) m, multiplier is

$$  
\left(\frac{10}{20}\right)^{1.2}  
\approx0.435;  
$$
This means farther returns remain dimmer because TVG has not completely compensated the range loss.

```python
import numpy as np


def apply_simple_tvg_residual(
    clean_intensity: np.ndarray,
    clean_range: np.ndarray,
    reference_range_m: float = 10.0,
    residual_exponent: float = 1.2,
) -> np.ndarray:
    safe_range = np.clip(clean_range, 0.5, None)

    gain = (
        reference_range_m / safe_range
    ) ** residual_exponent

    return clean_intensity * gain
```

If the real data suggests gain drift or automatic gain adjustment, evolve parameters smoothly:

 $$  
p_t =

0.98p_{t-1}  
+  
0.02\mu_p  
+  
\epsilon_t.  
$$

For example:

```python
def update_tvg_exponent(
    previous: float,
    mean: float = 1.2,
    persistence: float = 0.98,
    innovation_std: float = 0.02,
) -> float:
    return (
        persistence * previous
        + (1.0 - persistence) * mean
        + np.random.normal(0.0, innovation_std)
    )
```
