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

[The "less" important metrics are called "satisficing metric"](./2022-02-15-deep-learning-performance-metrics.markdown). They are usually the **first thing you should ** figure out when starting a project

But if you don't have a clear idea yet, define one and get stared. Later, refine it.

Then, worry **separately** about how to perform well on this metric.

### Bayes Optimal Error

Bayes Optimal Error is the error rate of best-possible error that can never be surpassed. A model's accuracy usually increases fast until it reaches "human level accuracy". After that, it will slow down as it approaches Bayes Optimal Error. Many times though, we **assume that human level accuracy is close to Bayes Optimal Error and use human level accuracy as a proxy to Bayes Optimal Error**

- This on the other hand, determines "what is human error?" An amateur human level performance, or expert? The Bayes error can only be lower than the expert, so which one do you think ;)

The gap between Bayes Optimal Error and the current training error is called **"avoidable bias"**. **Variance** is then the difference in training set and dev sets' error rates.

```
Human Level
^
|
avoidable bias
|
v
Training Error
^
|
variance
|
v
Dev error
```

If you have surpassed human level performance, you might be overfitting. So it's kind of hard to tell whether you should focus on reducing bias or variance. Structured data tasks, like ads recommendation, transit time predictions, loan approvals, are easier for machine learning systems because of the abundance of data. But for perception tasks such as vision or audio, they are harder for the relative lack of data and humans are usually pretty good at them.

## What To Do To Improve Performance

**First and foremost, what is worth our effort?** Do an error analysis on a small set of mislabelled data and get a sense of what the common errors are. It's probably worth it to count the number of mislabelled data of each category, then that could help you find if you need to focus on misclassification of dogs, great cats, etc.

**Second is to improve in the area we have chosen.** Usually so long as your system is worse than human performance, you might want to start thinking "why humans are doing better?"

If avoidable bias is high, maybe try:

- Train Longer
- Try better optimization algo (RMSprop, Momentum, Adam)
- Try a bigger model
- Another architecture
- Maybe decrease regularization (if variance is low)

If the variance is high:

- Try more data
- try regularization: L2, dropout, data augmentation

**Third, what if there is some mislabelling?** DL algorithms have some robustness to "random" mislablling at a small amount. So in error analysis, maybe it's a good idea to count:

- The number of mislabelling in the general subdataset we looked at. That gives us the overall mislabelling rate.
- The percentage of errors that were due to mislabelling. If this rate is high, maybe it'd be worth it to fix it.

Dev set is to tell which classifier A&B is better. If you fix labels in the dev set, fix some in the test set as well.

**Fourth, it's important to note that correcting wrong labels training set is a bit less important than correcting the dev/test set**, because your DL algorithm is somewhat robust to training set errors.

## Start Your Application Early, Then Iterate

Even after many years in speech recognition, Andrew still had some difficulties bringing up a speech recognition system that's super robust to noises. So, start your application early by:

- Setting up dev/test sets, metrics
- Start training a model, maybe two
- Use bias/variance analysis and error analysis to prioritize next step.

Most of the times, people overthink and build something too complicated

### What If Training Data Is A Different Distribution?

Say you have 1000 pictures of your cats, but you also have access to a 100000 cat images on the internet that look quite different from your cats. Then, what should you do?

Option 1 is to mix these pictures together and shuffle them, so you will have the same distribution across train/dev/test sets. However, your target distribution will be different. 

Option 2 is to have your test/dev sets only to be from the target distributions. You can add some of your cat pictures to your training set. This way. at least your target distribution will be what you want. **This is better for the long term.**