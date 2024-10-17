---
layout: post
title: Deep Learning - Strategies Part 1 Before Model Training
date: '2022-05-17 13:19'
subtitle: Orthogonalization
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Start Your Application Early, Then Iterate

Even after many years in speech recognition, Andrew still had some difficulties bringing up a speech recognition system that's super robust to noises. So, start your application early by:

- Setting up dev/test sets, metrics
- Start training a model, maybe two
- Use bias/variance analysis and error analysis to prioritize next step.

Most of the times, people overthink and build something too complicated. Below, I will walk through Error Metric Determination, and Data Preparation Before Model Development

## Error Metric

Don't stick to an metric that doesn't capture the characteristics of the system, or how well your application actually performs. Some examples include:

- User uploads pictures with less resolution while your model is trained with high resolution. In that case, include resolution in the evaluation metric
- Classifier needs to work with very important / sensitive data. Misclassifying that data comes with a high cost in reality.

[The "less" important metrics are called "satisficing metric"](./2022-02-15-deep-learning-performance-metrics.markdown). They are usually the **first thing you should** figure out when starting a project

But if you don't have a clear idea yet, define one and get stared. Later, refine it.

Then, worry **separately** about how to perform well on this metric.

## Data Preparation

There are optionally two/three sets:

- **Training set**
- **Holdout / dev set** (optional) is the unbias evaluation of the model fit for hyperparameter tuning. It can be also used for early stopping and regularization. So a validation set is often used in between epochs.
- **Test / validation set**: the final evaluation of the model's performance. The dataset should never have been seen in training.

Traditionally, the ratio of Training vs Hold out vs Validation is: 70%/20%/10%. For Big data: 98%/1%/1%

### Data Split Should Keep The Same Distribution

It is important that all datasets **come from the same distribution**. One way to do this is through **"stratified sampling"**. Stratified sampling is commonly used in surveys, where each subgroup (a.k.a strata) in a population is proportionately drawn from. That is, if we have a city with 60% population over 40, then in a survey for 100 people, we want to choose 60 people over 40.

If you have some new data to take into account, but they are pretty small. Augment them would be a good idea.

**What if training data is from a different distribution?**

If you have data from a distribution, it's not exactly problematic to add them to the training data. Adding them to the test / dev sets are problematic though.
Say you have 1000 pictures of your cats, but you also have access to a 100000 cat images on the internet that look quite different from your cats. Then, what should you do?

- Option 1 is to mix these pictures together and shuffle them, so you will have the same distribution across train/dev/test sets. However, your target distribution will be different.

- Option 2 is to have your test/dev sets only to be from the target distributions. You can add some of your cat pictures to your training set. This way. at least your target distribution will be what you want. **This is better for the long term.**

- Option 3 is to carve out a small portion of the training set and use it as a training-dev set. This way, you can ensure if there's overfitting in the training set. Otherwise, there might be a **data mismatch** problem where the dev set comes from a different distribution.

### Mislabelling

**what if there is some mislabelling?** DL algorithms have some robustness to "random" mislablling at a small amount. So in error analysis, maybe it's a good idea to count:

- The number of mislabelling in the general subdataset we looked at. That gives us the overall mislabelling rate.
- The percentage of errors that were due to mislabelling. If this rate is high, maybe it'd be worth it to fix it.
- Dev set is to tell which classifier A&B is better. If you fix labels in the dev set, fix some in the test set as well.
- **It's important to note that correcting wrong labels in training set is a bit less important than correcting the dev/test set**, because your DL algorithm is somewhat robust to training set errors.
