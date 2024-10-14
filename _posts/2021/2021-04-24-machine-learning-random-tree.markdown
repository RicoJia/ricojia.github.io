---
layout: post
title: Computer Vision - Random Tree
date: '2021-04-24 13:19'
subtitle: Random Tree
comments: true
tags:
    - Classical Machine Learning
---

## Introduction

Simple Example: Predicting Lottery Purchase with a Decision Tree

Imagine we have two pieces of information about a person:

- Age
- Income

The task is: we want to predict whether the person will buy a lottery ticket (yes/no).

| ID  | Age  | Income  | BuyLottery |
|-----|------|---------|------------|
| 1   | 25   | 35000   | 1          |
| 2   | 40   | 60000   | 0          |
| 3   | 22   | 25000   | 1          |
| 4   | 35   | 80000   | 0          |
| 5   | 28   | 45000   | 1          |
| 6   | 50   | 90000   | 0          |
| 7   | 19   | 30000   | 1          |
| 8   | 45   | 75000   | 0          |
| 9   | 30   | 52000   | 0          |
| 10  | 23   | 28000   | 1          |
| 11  | 33   | 49000   | 1          |
| 12  | 29   | 47000   | 1          |
| 13  | 52   | 100000  | 0          |
| 14  | 41   | 62000   | 0          |
| 15  | 27   | 42000   | 1          |

1. Step 1: Root Node (Starting Decision):

    The tree begins with the entire dataset at the root.
    We choose the feature that best splits the data. Let’s say the most important factor is age.

2. Step 2: Split the Data:

    If yes (age < 30), go to the left branch.
    If no (age ≥ 30), go to the right branch.

3. Step 3: Further Decisions:

    In the left branch (age < 30), the tree now looks at income to refine the decision:
        If income < $50,000, the prediction might be "Buys Lottery".
        If income ≥ $50,000, the prediction might be "Does Not Buy Lottery".

    In the right branch (age ≥ 30), the tree might not need to split further

    The prediction might simply be "Does Not Buy Lottery" for all people in this group.

```
          Is Age < 30?
           /       \
      Yes            No
     /                \
Is Income < $50k?   Does Not Buy
   /    \
Yes    No
```

How do we determine the best value of each split? Define Binary Cross-Entropy (BCE) as the loss function:

$$
\begin{gather*}
L = -\sum y_ilog(\hat{y_i}) + (1-y_i)log(1-\hat{y_i})
\end{gather*}
$$

Then we can try all values at each split and based on the losses, we can find the best one.