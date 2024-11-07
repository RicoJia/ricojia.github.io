---
layout: post
title: Deep Learning - Performance Metrics
date: '2022-02-15 13:19'
subtitle: mean Average Precision (mAP), ROC Curve
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Terminology

### Area Under Curve

Area Under Curve = AUC.

### True Positives, False Positives, True Negatives, False Negatives

Suppose a class has 10 students, 5 boys and 5 girls. You use a machine to find the girls, and the machine returns the following results:

```
| Boy | Girl | Girl | Boy | Girl | Boy |
```

Mathematically,

- Positives: The 6 results returned by the machine (these are called Positives because the machine identified them as "girls").
- Negatives: The 4 remaining students who were not returned by the machine are considered Negatives.

So,

- **True Positives**: These are the items (students) that the machine classified as "girls" and are actually girls. In this case, the machine correctly identified 3 out of the 5 girls:

- **False Positives (FP)**: These are the items the machine classified as "girls," but they are actually boys. In this case, the machine wrongly classified 3 boys as "girls".

- **False Negatives (FN)**: These are the actual girls that the machine missed (classified as "boys"). Since there are 5 actual girls and the machine found 3 of them, it missed 2 girls

- **True Negatives (TN)**: These are the boys that the machine correctly classified as "boys." Since the class has 5 boys in total, and the machine incorrectly labeled 3 as girls, it correctly identified 2 boys as boys:

### Confusion Matrix

A confusion matrix shows how often an example whose label is one class ("actual" class) is mislabeled by the algorithm with a different class ("predicted" class). This measures how "confused" a binary classifier is in predictions

| | Predicted Positive | Predicted Negative |
| -------- | -------- | -------- |
| Actual Positive | |
| Actual Negative | |

## ROC (Receiver Operating Characteristic Curve)

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/929f74c6-ef03-42f4-9ecf-8b14f63920e8" height="300" alt=""/>
        <figcaption><a href="https://medium.com/@ilyurek/roc-curve-and-auc-evaluating-model-performance-c2178008b02">Source: </a></figcaption>
    </figure>
</p>
</div>

ROC curve measures the performance of a binary classifier at different decision threshold (TODO what is decision threshold?).

The y axis is True Positive Rate: "what percentage of the positives have you classified" This is the same as "recall" as below

$$
\begin{gather*}
\frac{TP}{TP + FN}
\end{gather*}
$$

The x axis is False Positive Rate: "what percentage of the negatives have you classified".

$$
\begin{gather*}
\frac{FP}{FP + TN}
\end{gather*}
$$

A decision threshold of a binary classifier is the threshold of probability or confidence above which a class can be identified as true. **A ROC curve is made when the decision threshold is varied.**

- A random classfier would have equal FPR and TPR rates.

Why it's called **ROC**: In World War II, the ROC curve was used by the radar operators to detect enemies. Receiver means "radar receiver equipment", and the curve measures the radar receiver's ability to distinguish enemies, out of all the enemies, versus false signals like flying objects like birds out of all non-enemy flying objects.

In general, people uses AUC to describe a ROC curve. **Pitfalls** of ROC include:

- In imbalanced datasets, when positives are rare, even good models might easily have a low TPR, which results in a low ROC value. An example is medical imaging where the positive (disease) could be rare.
- So, negative predicted value should be measured along with ROC

$$
\begin{gather*}
NPV = \frac{TN}{TN + FN}
\end{gather*}
$$

- Precision should be measured along with ROC, see below

## mean Average Precision (mAP)

### Precision (查准率) and Recall (查全率)

The precision is: 3/6 = 0.5. Interpretation: It is "#accurate items / #items being returned"
The recall is: 3/5 = 0.6. Interpretation: This is "#accurate items / #total number of target items", like how many of the total items you've "recalled" into your search.

$$
\begin{gather*}
\text{Precision} = \frac{TP}{TP+FP}
\\
\text{Recall} = \frac{TP}{TP+FN}
\end{gather*}
$$

#### AP Definition in Image Detection

In the case of object detection, **True Positive is the intersecting area, False Positive is the rest of the detected box, False Negative is the rest of the object box.**

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/034d4ba7-d217-4a8c-a92a-51d8506b1120" height="200" alt=""/>
        <figcaption><a href="https://www.kdnuggets.com/2020/08/metrics-evaluate-deep-learning-object-detectors.html">Source</a></figcaption>
    </figure>
</p>
</div>

### Average Precision

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/671611c0-f051-4d0c-9cce-37c2759a2dc5" height="300" alt=""/>
    </figure>
</p>
</div>

In this real world example, we have found 7 bounding boxes. We sort them by their **confidence**

| id | confidence | IoU  | Precision | Recall |
|----|------------|------|-----------|--------|
| 2  | 0.99       | 0.7  | 0.94      | 0.80   |
| 1  | 0.96       | 0.9  | 0.96      | 0.95   |
| 3  | 0.95       | 0.95 | 1.0       | 0.95   |
| 5  | 0.93       | 0.85 | 1.0       | 0.90   |
| 4  | 0.87       | 0.8  | 1.0       | 0.80   |
| 6  | 0.67       | 0.45 | 0.91      | 0.70   |
| ~~7~~  | ~~0.33~~       | ~~0~~    | ~~0.0~~       | ~~0.0~~|

**Note that the IoU of box 7 is 0. We usually discard boxes with IoU lower than 0.5.**. So we discard that one.

Average Precision (AP) is the **AUC of the precision-recall curve.** In real life, we want a fast way to calculate an approximate of that. Because precision and recall are values in `[0, 1]`, the final AP is also `[0, 1]`

#### PASCAL VOC (Visual Object Classes) 11-Point Interpolation, Pre-2010

PASCAL VOC is a common dataset for object detection. Pre-2010, AP is calculated by taking the mean of the highest value in intervals `[0.0, 0.1, ... 1.0]`. This method is called **Interpolated AP**

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/762ae2ab-9756-4be0-929c-ac935ef3b747" height="300" alt=""/>
        <figcaption><a href="">Source: </a></figcaption>
    </figure>
</p>
</div>

$$
\begin{gather*}
AP = \frac{1}{11} \sum_{r \in [0.0, 0.1, 0.2...]}(P_{interpolated}(r))
\end{gather*}
$$

where

$$
\begin{gather*}
P_{interpolated}(r) = max(P_{r > \tilde{r}}(r))
\end{gather*}
$$

so when `r=0.9`, $$P_{interpoloated}=max(0.96,1.0, 1.0) = 1.0$; `r=0.8, ...`,  will get the same $P_{interpoloated}$. The final AP is `(1.0 + 1.0 + ...)/11=1.0`. The intuition comes from that normally, **as recall goes up, precision goes down.**

#### PASCAL VOC (Visual Object Classes) Post-2010

$$
\begin{gather*}
AP = \sum_r (r_{n+1}-r_{n})P_{interpolated}(r_{n+1})
\\
P_{interpolated}(r_{n+1}) = max(P_{r \in \tilde{r}}(r))
\end{gather*}
$$

So one difference from PASCAL VOC pre-2010 is we are calculating for **every recall level.** instead of calculating for the 10 fixed intervals.

#### COCO Dataset mAP

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/b55a7b51-97e2-42b8-9699-7004b5a518b5" height="300" alt=""/>
        <figcaption><a href="https://cocodataset.org/#detection-eval">Source </a></figcaption>
    </figure>
</p>
</div>

COCO has its own metrics. The primary metric is AP `AP[.50:.05:.95]`. They are APs of images with IoUs from `[0.5, 0.95]` with 0.05 interval sizes. So they are:

$$
\begin{gather*}
AP = \frac{1}{10} \sum_{n=0}^{n=9}AP[0.5, 0.5+0.05n]
\end{gather*}
$$

And each $AP$ is taken over 101 recall levels.

### Mean Average Precision (mAP)

Finally, mean Average Precision mAP is to take the mean of Average Precion across all classes

$$
\begin{gather*}
mAP = \frac{\sum_C AP(c)}{C}
\end{gather*}
$$

## F1 Score

Having a single-value metric makes evaluation intuitive. An F1 score is the "harmonic average" of Precision, and Recall

$$
\begin{gather*}
F1 = \frac{2}{\frac{1}{precision} + \frac{1}{recall}}
\end{gather*}
$$

### Satisficing Metric

Satisficing here means "satisfying a certain metric suffices". It's a kind of metric that we set a minumum requirement for, but do not care so much afterwards. For example, in a classifier, as long as recall is over 90%, we don't care about it as much; or in a recommendation system, we set a minimum for speed, but after that we care a lot more on the accuracy.

## References

[1] [深入了解平均精度(mAP)：通过精确率-召回率曲线评估目标检测性能](https://cloud.tencent.com/developer/article/2318730)
[2] [mean Average Precision (mAP) — 評估物體偵測模型好壞的指標](https://medium.com/lifes-a-struggle/mean-average-precision-map-%E8%A9%95%E4%BC%B0%E7%89%A9%E9%AB%94%E5%81%B5%E6%B8%AC%E6%A8%A1%E5%9E%8B%E5%A5%BD%E5%A3%9E%E7%9A%84%E6%8C%87%E6%A8%99-70a2d2872eb0)
