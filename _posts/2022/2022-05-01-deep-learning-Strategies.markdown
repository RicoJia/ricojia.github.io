---
layout: post
title: Deep Learning - Strategies
date: '2022-05-17 13:19'
subtitle: Orthogonalization
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Orthogononalization

Orthogonalization in ML means designing a machine learning system such that different aspects of the model can be adjusted independently. This is like "orthogonal vector" so that they are independent from each other.

```
training set -> dev set -> test set
```

In general, first, get your training set accuracy good. Some knobs there include bigger network, different optimizer, etc.
Then, if dev set performance is not very good, tune regularization.
Then, if test set performance is not very good either, maybe have a larger dev set.

Early stopping is less "orthogonal" in a sense that it simultaneously affects two things: potentially lower performance on the training set, and improving on the test set.

## Error Metric

Don't stick to an metric that doesn't capture the characteristics of the system, or how well your application actually performs. Some examples include:

- User uploads pictures with less resolution while your model is trained with high resolution. In that case, include resolution in the evaluation metric
- Classifier needs to work with very important / sensitive data. Misclassifying that data comes with a high cost in reality.

But if you don't have a clear idea yet, define one and get stared. Later, refine it.

Then, worry **separately** about how to perform well on this metric.

### Bayes Optimal Error

Bayes Optimal Error is the error rate of best-possible error that can never be surpassed. A model's accuracy usually increases fast until it reaches "human level accuracy". After that, it will slow down as it approaches Bayes Optimal Error. Many times though, we **assume that human level accuracy is close to Bayes Optimal Error and use human level accuracy as a proxy to Bayes Optimal Error**

- This on the other hand, determines "what is human error?" An amateur human level performance, or expert? The Bayes error can only be lower than the expert, so which one do you think ;)

The gap between Bayes Optimal Error and the current training error is called **"avoidable bias"**. **Variance** is then the difference in training set and dev sets' error rates.

Usually so long as your system is worse than human performance, you might want to start thinking "why humans are doing better?"

If you have surpassed human level performance, you might be overfitting. So it's kind of hard to tell whether you should focus on reducing bias or variance. Structured data tasks, like ads recommendation, transit time predictions, loan approvals, are easier for machine learning systems because of the abundance of data. But for perception tasks such as vision or audio, they are harder for the relative lack of data and humans are usually pretty good at them.