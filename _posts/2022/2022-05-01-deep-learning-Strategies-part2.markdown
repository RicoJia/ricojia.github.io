---
layout: post
title: Deep Learning - Strategies Part 2 Training And Tuning
date: '2022-05-17 13:19'
subtitle: Bias And Variance, And Things To Try For Performance Improvement From My Experience
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

## Bias And Variance

```
Bayes Optimal Accuracy

    ^
    |
    Unknown, use human accuracy as a proxy
    |
    v

Human Accuracy
    
    ^
    |
    Avoidable Bias 
    |
    v

Training Set Accuracy

    ^
    |
    Variance
    |
    v

Train-Dev Set Accuracy
    
    ^
    |
    Data Mismatch
    |
    v

Dev Accuracy
    ^
    |
    Degree of overfitting
    |
    v

Test Set Accuracy
```

- Bayes Optimal Error is the best-possible error that can never be surpassed. A model's accuracy usually increases fast until it reaches "human level accuracy". After that, it will slow down as it approaches Bayes Optimal Error. Many times though, we **assume that human level accuracy is close to Bayes Optimal Error and use human level accuracy as a proxy to Bayes Optimal Error**. This on the other hand, determines "what is human error?" - an amateur human level performance, or expert? The answer is "we always go for the best one" ;)

## Do An Error Analysis

1. **First and foremost, what is worth our effort?** Do an error analysis on a small set of mislabelled data and get a sense of what the common errors are. It's probably worth it to count the number of mislabelled data of each category, then that could help you find if you need to focus on misclassification of dogs, great cats, etc.

2. **Second is to improve in the area we have chosen.** Usually so long as your system is worse than human performance, you might want to start thinking "why humans are doing better?"

## Scenarios From Error Analysis And What To Do

This section is from my own training experience

- Avoidable bias is very high:
  - If training set accuracy does not increase
        1. Did you set backpropagation right?
            - Made a mistake where I called

                ```python
                for batch in dataloader:
                    output = model(batch)
                    loss = criterion(output)
                loss.backward()
                # I forgot optimizer.step(), too, rookie mistake :/
                ```

        2. Try training Longer.
            - This is assuming that your loss could still decrease with a learning rate scheduler
        3. Try a better optimization algo (RMSprop, Momentum, Adam)
        4. Try decreasing regularization (if variance is low)
        5. Try a bigger model
        6. Try another architecture

  - If training set accuracies have increased but capped at a certain value.
    - Debug from the end result from prediction. Does the end result match with what you expected: shape, datatype, data value, etc.?

  - Loss could be defined better?
    - In Image Segmentation tasks, `cross entropy` loss does not handle data inbalance well (e.g., too many background pixels). `Dice Loss` or `Tversky Loss` could be better.

  - Lastly, surpassing human level performance is hard. Structured data tasks like ads recommendation, transit time predictions, loan approvals, are easier for machine learning systems because of the abundance of data. But perception tasks such as vision or audio are harder for the relative lack of data and humans are usually pretty good at them.

- What if the variance is high:
    1. Try with more data.
    2. Try with more regularization: L2, dropout, data augmentation, adding momentum if you are using RMSProp.

- What if data mismatching is significant:
  - Try artificial data synthesis. For example, in speech recognition, superpositioning car noise on human speech is a great boost.

- If you are overfitting. it's kind of hard to tell whether you should focus on reducing bias or variance.

### Simulated Scenarios (From Coursera)

You are employed by a startup building self-driving cars. You are in charge of detecting road signs (stop sign, pedestrian crossing sign, construction ahead sign) and traffic signals (red and green lights) in images. The goal is to recognize which of these objects appear in each image.

1. Your 100,000 labeled images are taken using the front-facing camera of your car. This is also the distribution of data you care most about doing well on. You think you might be able to get a much larger dataset off the internet, which could be helpful for training even if the distribution of internet data is not the same. Suppose that you came from working with a project for human detection in city parks, so you know that detecting humans in diverse environments can be a difficult problem. What is the first thing you do? Assume each of the steps below would take about an equal amount of time (a few days).

- A. Train a basic model and proceed with error analysis.
- B. Leave aside the pedestrian detection, to move faster and then later solve the pedestrian problem alone.
- C. Spend a few days collecting more data to determine how hard it will be to include more pedestrians in your dataset.
- D. Start by solving pedestrian detection, since you already have the experience to do this.

**Answer**: Though C might be true, it's always better to start the iteration earlier.

2. If some pictures were collected with labels that only contain `[car, pedestrian]`, without the remaining classes, what should you do?

- A. Make the missing labels as 1 (detected)
- B. Calculate loss that only considers `[car, pedestrian]`

**Answer:** A is more "false positive" prone. B is more reasonable to use.

3. Your training set error is quite much higher than that of humans. So:

- A. You have a variance issue
- B. You have an avoidable issue

**Answer:** A.

4. If your errors due to mislabelling is 4.2%, does that mean your accuracy will increase by 4.2% after we fix them?

**Answer:** No. That's the ceiling. There might be other issues with the associated data.

5. If your errors are mostly due to occlusion. Should you work on it immediately?

**Answer:** Maybe not. We have to weight how easy it is to get more images for that.
