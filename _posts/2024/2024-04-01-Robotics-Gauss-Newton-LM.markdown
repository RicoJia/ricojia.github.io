---
layout: post
title: Robotics - Gauss Newton (GN) and Levenberg Marquardt (LM) Optimizers
date: '2024-07-11 13:19'
subtitle: Newton's Method, GN, LM optimizers
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
---

## Newton's Method

### Solving An Equation

To find an arbitrary equation's root $f(x) = 0$,

- We start from an arbitrary point $x_0$ that's *hopefully* close to the solution, *$x_s$*
- **The main idea of Newton's method is, we draw a tangent line at $x_0$, this line will cross `y=0` at $x_1$. It's most likely that this point is closer to $x_s$ than $x_0$**. So, this is to solve $x_1f(x_0) - x_0f'(x_0) + f(x_0) = 0$, and we get $x_1 = x_0 - \frac{f(x_0)}{f'(x_0)}$

So at each iteration, the newer estimate is:

$$
\begin{gather*}
\begin{aligned}
& x_{n+1} = x_{n} - \frac{f(x_0)}{f'(x_0)}
\end{aligned}
\end{gather*}
$$

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/eaa327b9-c0d0-40c7-a8f0-b5ec55f7fca3" height="300" alt=""/>
       </figure>
    </p>
</div>

Some caution-worthy notes are:

- $f'(x_0)$ should be non-zero. Otherwise, the estimate will NOT move
- In vanilla gradient descent, we only find the gradient and issue an update $\Delta x = \lambda g$ using a fixed step size $\lambda$. However, in Newton's method, we not only follow the gradient direction, but we also "solve" for the step size. That makes Newton's method faster.

### Example: Solve For Square-Roots

If we want find $\sqrt{m}$, we can format this problem to solving:

$$
\begin{gather*}
\begin{aligned}
& f(x) = x^2 - m = 0

\\
& \rightarrow x_{n+1} = x_{n} - \frac{x_n^2 - m}{2 x_n} = \frac{x_n}{2} + \frac{m}{2 x_n}
\end{aligned}
\end{gather*}
$$

```cpp
const double EPS = 0.0001; 
double sqrt(double m){
    if (m == 0) return m;
    double x = m;
    double last_res;
    do {
        last_res = x;
        x = x / 2.0 + m / (2 * x);
    } while (abs(last_res - x) > EPS);
    return x;
}
```

[Faster version (TODO), inspired by John Carmack](https://en.wikipedia.org/wiki/Fast_inverse_square_root)

```cpp
double sqrt_c(float x) { 
    if(x == 0) return 0; 
    float result = x; 
    float xhalf = 0.5f*result; 
    int i = *(int*)&result; 
    // 1048009350
    i = 0x5f375a86- (i>>1); // what the fuck? 
    result = *(float*)&i; 
        cout<<result<<endl; //0.241556
    result = result*(1.5f-xhalf*result*result); // Newton step, repeating increases accuracy 
    result = result*(1.5f-xhalf*result*result); 
    return 1.0f/result; 
}
```

### Newton's Method for Optimization

When optimizing `f(x)`, again we start from an arbitrary point $x_0$. If we can achieve the optimum at $x_0 + \Delta x$:

We think of it as:

$$
\begin{gather*}
\begin{aligned}

\text{Taylor Expansion:}
\\
& f(x + \Delta x) \approx f(x_0) + f'(x_0) \Delta x + \frac{1}{2} (f''(x_0))^2 \Delta x^2

\\
\text{Optimum: }
\\
& \frac{\partial f(x_0 + \Delta x)}{\partial \Delta x} = 0

\\
& \rightarrow f(x + \Delta x)' = f'(x_0) + f''(x_0) \Delta x = 0
\\
& \rightarrow  \Delta x = -\frac{f'(x_0)}{f''(x_0)}
\\
& \rightarrow x_1 = x_0 + \Delta x = x_0 -\frac{f'(x_0)}{f''(x_0)}
\end{aligned}
\end{gather*}
$$

So iteratively, at timestep `n`, we have:

$$
\begin{gather*}
\begin{aligned}
& x_{n+1} =  x_n -\frac{f'(x_n)}{f''(x_n)}
\end{aligned}
\end{gather*}
$$

See how Newton's method has similar forms for both equation solving and optimization? The better the function `f` is approximated by Taylor Expansion, the faster we converge.

For higher dimensions, **if `f` is a scalar-valued function, and `x` is `[m,1]`** we have:

$$
\begin{gather*}
\begin{aligned}
& x_{n+1} =  x_n - J(x_n) H(x_n)^{-1}
\end{aligned}
\end{gather*}
$$

- J is `[1, m]`, H is `[m,m]`

## Gauss-Newton Optimization

In Gauss Newton, we specifically look at minimizing a least squares problem. Assume we have a:

- scalar-valued cost function $c(x)$,
- vector-valued function: $f(x)$, `[m, 1]`
- Jacobian $J_0$ at $x_0$ is consequently `[m, n]`
- Hessian $H$ is $D^2c(x)$. It's approximated as $J^T J$

$$
\begin{gather*}
\begin{aligned}
& c(x) = |f(x)^2|
\\
& x* = argmin(|f(x)^2|)

\\
\text{First order Taylor Expansion:}
\\
& argmin_{\Delta x}(|f(x + \Delta x)^2|)
\\
&= argmin_{\Delta x}[(f(x_0) + J_0 \Delta x)^T (f(x_0) + J_0 \Delta x)]

\\
& = argmin_{\Delta x}[f(x_0)^T f(x_0) + f(x_0)^T J_0 \Delta x + (J_0 \Delta x)^T f(x_0) + (J_0 \Delta x)^T (J_0 \Delta x)]

\\
& = argmin_{\Delta x}[f(x_0)^T f(x_0) + 2 f(x_0)^T J_0 \Delta x + (J_0 \Delta x)^T (J_0 \Delta x)]
\end{aligned}
\end{gather*}
$$

Take the derivative of the above and set it to 0, we get

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial f(x + \Delta x)^2}{\partial \Delta x} = 2J_0^T f(x_0) + [(J_0^TJ_0) + (J_0^TJ_0)^T]\Delta x

\\
& = 2J_0^T f(x_0) + 2(J_0^TJ_0) \Delta x

\\
& = 0
\end{aligned}
\end{gather*}
$$

So we can solve for $\Delta x$ with $H = J_0^TJ_0$, $b = - J_0^T f(x_0)$:

$$
\begin{gather*}
\begin{aligned}
& (J_0^TJ_0) \Delta x = - J_0^T f(x_0)
\\
& \rightarrow H \Delta x = g
\end{aligned}
\end{gather*}
$$

- Note: because $J_0$ may not have an inverse, here we cannot multiply $J_0^{-1}$ to eliminate $J_0^T$
- In fact, to $\Delta x$ is available if and only if $H$ is **positive definite**.
- In least square, $f(x)$ is a.k.a residuals. Usually, it represents the **error between a data point and from its ground truth**.

In SLAM, we always frame this least squares problem with `e = [observered_landmark - predicted_landmark]` at each landmark. So all together, we want to **minimize the total least squares of the difference between  observations and predictions.** In the meantime, at each landmark, there is an error covariance, so all together, there's an error matrix $\Sigma$. Here in cost calculation, we take $\Sigma^{-1}$ so the **larger the error covariance, the lower the weight the corresponding difference gets.**

With $e(x + \Delta x) \approx e(x) + J \Delta x$,

$$
\begin{gather*}
\begin{aligned}
& x* = argmin(|e^T \Sigma^{-1} e|)
\\
& \rightarrow argmin_{\Delta x}(|e(x + \Delta x)^T \Sigma^{-1} e(x + \Delta x)|)
\\
& \text{similar steps as above ... }
\\
& \rightarrow (J_0^T \Sigma^{-1}J_0) \Delta x = - J_0^T \Sigma^{-1} f(x_0)
\end{aligned}
\end{gather*}
$$

Using Cholesky Decomposition, one can get $\Sigma^{-1} = A^T A$. Then we can write the above as

$$
\begin{gather*}
\begin{aligned}
& ((AJ_0)^T (AJ_0)) \Delta x  = - (AJ_0)^T Af(x_0)
\end{aligned}
\end{gather*}
$$

For a more detailed derivation, [please see here](./2024-07-11-rgbd-slam-bundle-adjustment.markdown)

## Levenberg-Marquardt (LM) Optimization

Again, **Taylor expansion** works better when $\Delta x$ is small, so the function can be better estimated by it. So, similar to regularization techniques on step sizes in deep learning, like L1, L2 regularization, we can regularize the step size, $\Delta x$

$$
\begin{gather*}
\begin{aligned}
& (H + \mu I_H) \Delta x = -J_0^Tf(x_0) = g
\end{aligned}
\end{gather*}
$$

Intuitively,

- as $\mu$ grows, the diagonal identity matrix $\mu I_H$ grows, so $H + \mu I_H \rightarrow \mu I_H$. So, $\Delta x \approx (H + \mu I_H)^{-1}g = \frac{g}{\mu}$, which means $\Delta x$ grows smaller. In the meantime, $\Delta x$ will be similar to that in gradient descent.
- as $\mu$ becomes smaller, $\Delta x$ will become more like Gauss-Newton. However, due to $\mu I_H$, $(H + \mu I_H)$ is positive semi-definite, which provides more stability for solving for $\Delta x$.

## References

<https://scm_mos.gitlab.io/algorithm/newton-and-gauss-newton/>
