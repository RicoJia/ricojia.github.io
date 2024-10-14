---
layout: post
title: Deep Learning - Ensemble
date: '2022-03-03 13:19'
subtitle: Ensemble
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Ensemble

An ensemble is a group of models (a.k.a base learners, weak learners) that are trained and combined to have better prediction, increased stability, and improved generalization compared to individual models.

### Methods

### Bagging (Bootstrap Aggregating)

Train multiple models independently using random subsets of data. Example: Random Forest. Advantage: reduces variance and helps prevent overfitting

#### Boosting

Train models sequentially, each tries to minimize error from the previous one. Example: Adaboost, Gradient Boosting machines like XGBoost. Advantages:?

#### Stacking

Stacked generatlization creates a combined "meta-model" to learn how to best combine their predicitons?

#### Voting and Average

For classification, use majority voting. for regression, average the predictions
