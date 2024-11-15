---
layout: post
title: Deep Learning - Strategies Part 1 Before Model Training
date: '2022-05-17 13:19'
subtitle: Error Metrics, Data Preparation Principles, Transfer Learning, Multi-Task Learning
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Start Your Development Early, Then Iterate

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

### Choice of Metrics

- For tasks with a lot of backgrounds, accuracy is not the best metric. F1 Score, precision & recall could be better scores. For example, [speech recognition](./2022-04-02-deep-learning-speech-recognition-hands-on.markdown)

## Data Preparation

There are optionally two/three sets:

- **Training set**
- **Holdout / dev set** (optional) is the unbias evaluation of the model fit for hyperparameter tuning. It can be also used for early stopping and regularization. So a validation set is often used in between epochs.
- **Test / validation set**: the final evaluation of the model's performance. The dataset should never have been seen in training.

Traditionally, the ratio of Training vs Hold out vs Validation is: 70%/20%/10%. For Big data: 98%/1%/1%

But you can also have a **Train-Dev set** to ensure about performance on variance.

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

## Transfer Learning

If you are working on cancer image detection, you can probably take a pre-trained image recognition model, then do transfer learning. In that case, the pre-trained model already recognizes low level features such as corners and edges. If you:

- have **a large training dataset**, you might be able to retrain the entire model. This is called **"fine-tuning"**
- have a **small training dataset**, you could train just the last few custom layers. This is called **feature extraction**

However, if your pre-trained model is NOT trained on a large enough dataset, (say relatively small dataset), then the pretaining is probably not super beneficial.

### Multi-task Learning

When you work on a multi-object detection task, a single image could have multiple objects. Now, you have two options:

1. Train a network for each class
2. Train a single network for all classes at once.

Option 2 is called multi-task learning. Rich Caruana stated that multi-task learning won't hurt learning performance as long as the network is big enough. Multi-task learning will benefit most if **the tasks are related** so information learned from one task can be shared to others, such as learning a car, a pedestrian, etc. One note of caution is, maybe sticking to transfer learning is the best if your training data is not much.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/9aaabb88-f48a-4266-8e66-e3f1c12f4da4" height="300" alt=""/>
        <figcaption><a href="https://www.geeksforgeeks.org/introduction-to-multi-task-learningmtl-for-deep-learning/">Source</a></figcaption>
    </figure>
</p>
</div>

The final loss function for a batch of size $i$, $j$ classes is:

$$
\begin{gather*}
Loss = \sum_i \sum_j L(y_j^{(i)}, \hat{y}_j^{(i)})
\end{gather*}
$$

$L$ can be BCE Loss for each class.

### End-To-End Learning

End-to-end (e2e) learning is a hot topic, who doesn't want "one model that handles all"?

Speech Recognition is one good example, where the entire `audio -> features -> phonemes (distinc units of sounds like 'cuh'-'a'-'th' for cat) -> words -> transcript` chain is learned.

- Though Andrew argues that phonemes is a human-created artifact that's easy for humans to understand, but not by machines.

However, e2e learning depends on the amount of **data** that you have. Some counter examples include:

- Face recognition leverages a two stage process "face-localization -> face comparison". The reason is these two steps are relatively easier, and either has lots of data but not combined.
- Hand xray->age recognition for medical uses: the two step process `Image -> bones -> age` has more data than the two steps combined.

More specifically, the `x->y` mapping is crucial in deciding whether or not to use e2e learning. In autonomous vehicles' workflows,

```
Image, Lidar, IMU -> Car, pedestrian -> Route -> Steering
```

`Image, Lidar, IMU -> Car` can be handled by DL pretty well. But the motion planning algorithms and steering control requires a lot of data that has the above mapping. However, currently (as of 2024) we still don't have good real-world data on this front yet.

## References

[1] [Rich Caruana's Ph.D thesis, Carnegie Melon University, 1997](http://reports-archive.adm.cs.cmu.edu/anon/1997/CMU-CS-97-203.pdf)
