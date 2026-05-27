---
layout: post
title: YOLO V1-V7 Models Review
date: 2026-04-14 13:19
subtitle: ""
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine-Learning
---

## Important Constructs

### Anchors

Each feature-map location = one grid cell / spatial position.
At each location, the detector places K predefined anchors. An anchor is a pre-chosen bounding box shape used as a starting guess. Earlier models like YOLOv3 predicts boxes at multiple grid scales, and each grid cell has anchor-based predictions.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://miro.medium.com/1*Sstv9IO76e5yRA9VKSVC5w.png" height="300" alt=""/>
    </figure>
</p>
</div>

### mAP@0.50:0.95 Metric

For each IoU threshold from 0.50 to 0.95, we decide whether each predicted bbox is a true positive by checking whether its IoU with a matched ground-truth box is at least that threshold. Then we sort predictions by confidence, compute the precision-recall curve, calculate AP as the area under that curve, and finally average AP across all IoU thresholds and classes.

- Collect all predictions for that class.
- Sort them from highest confidence to lowest confidence.
- For a fixed IoU threshold, mark each prediction as TP or FP.
- As you move down the confidence-ranked list, compute precision and recall at each step.
- The area under that precision-recall curve is AP.
- Repeat for each IoU threshold.
- Repeat for each class.
- Average across thresholds and classes.

---

## YOLOv1 (Joseph Redmon, Ali Farhadi, 2016)

[Paper](https://arxiv.org/pdf/1506.02640)

Before YOLOv1, CNN-based detection methods such as R-CNN and Fast R-CNN were prevalent. These methods first proposed bounding boxes, then classified objects inside them, which was relatively slow.

YOLOv1 used 24 convolutional layers and 2 fully connected (FC) layers. The architecture was inspired by GoogLeNet: convolutional layers learn features, while FC layers output confidence and bounding box coordinates. Bounding box detection and classification are learned jointly, which significantly improves speed.

![](https://i.postimg.cc/LXFVP4Xf/Screenshot-from-2026-05-21-10-55-50.png)

In particular, the bounding box refinement step is ultimately a regression. Its input is a bounding box proposal, e.g., `P=(Px​,Py​,Pw​,Ph​)`. The model will output a delta to this prediction: `[delta_x, delta_y, delta_w, delta_h]`.

Additionally, YOLO used:

- Leaky ReLU: like ReLU, but allows a small negative output when the input is negative. This avoids the **dying ReLU problem**. With vanilla ReLU, negative inputs become exactly 0; if a neuron keeps receiving negative inputs, its gradient can become zero and it stops learning.
- Dropout: after the first FC layer, dropout is enabled so randomly selected neurons are turned off. This prevents the network from relying too heavily on a narrow pathway and encourages more robust feature learning.
- Data augmentation.

![](https://i.postimg.cc/TYHYQCD0/Screenshot-from-2026-05-21-11-21-10.png)

YOLOv1 divides the input image into an $S \times S$ grid. The cell containing an object's center is responsible for detecting that object. Each predicted bounding box has a confidence score, and IoU (Intersection over Union) is used to filter poor predictions.

Limitations:

- Each grid cell can predict only one class.
- The loss for small boxes is treated similarly to large boxes, which can hurt localization quality for small objects.
- Localization errors are common.

## [YOLOv2](https://arxiv.org/pdf/1612.08242) (Joseph Redmon, Ali Farhadi, 2017)

Also known as YOLO9000, YOLOv2 can detect about 9000 classes.

```python
predictions = model(image)  # many grid-cell / anchor predictions

if sample.type == "detection":
    matched_preds = assign_predictions_to_ground_truth_boxes(predictions, gt_boxes)

    loss_box = box_loss(matched_preds.box, gt_boxes)
    loss_obj = objectness_loss(predictions.obj, gt_objectness)
    loss_cls = wordtree_class_loss(matched_preds.cls, gt_classes)

elif sample.type == "classification":
    selected_pred = choose_prediction_most_responsible_for_class(
        predictions,
        image_level_class
    )

    loss_box = 0.0       # no bbox supervision
    loss_obj = 0.0       # usually ignored
    loss_cls = wordtree_class_loss(
        selected_pred.cls,
        image_level_class
    )

loss = loss_box + loss_obj + loss_cls
loss.backward()
```

The biggest change is joint training on COCO and ImageNet. COCO has bounding boxes and class labels but fewer classes. For a detection image (for example, from COCO), YOLO9000 trains with `L = L_box + L_class + L_confidence` on `image + bbox + class label`.

When training with ImageNet classification data, the model still outputs many bounding boxes, but only class loss is computed. During backpropagation, neurons not involved in classification receive close-to-zero updates.

One note: we do not pad outputs with fake zero bounding boxes, because that would teach the model incorrect behavior.

Another idea is to infer subcategory probabilities. For example, suppose we have a golden retriever image from ImageNet, while COCO only has a generic `dog` class. YOLO9000 uses conditional probabilities:

```
P(Norfolk terrier)
=
P(animal | object)
× P(dog | animal)
× P(terrier | dog)
× P(Norfolk terrier | terrier)
```

This is done with WordTree. For a COCO output class like `dog`, logits can be interpreted along the taxonomy path.

```python
class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.logit_index = None  # index of this node in the model output

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


# Build tree
object_node = Node("object")

animal = Node("animal")
vehicle = Node("vehicle")
object_node.add_child(animal)
object_node.add_child(vehicle)

dog = Node("dog")
cat = Node("cat")
animal.add_child(dog)
animal.add_child(cat)

terrier = Node("terrier")
shepherd = Node("shepherd")
dog.add_child(terrier)
dog.add_child(shepherd)

norfolk = Node("Norfolk terrier")
yorkshire = Node("Yorkshire terrier")
terrier.add_child(norfolk)
terrier.add_child(yorkshire)

german = Node("German shepherd")
shepherd.add_child(german)

car = Node("car")
boat = Node("boat")
vehicle.add_child(car)
vehicle.add_child(boat)

prediction = {
    "box": box_pred,                  # x, y, w, h
    "objectness": objectness_logit,   # object confidence
    "class_logits": class_logits,     # WordTree logits
}

def conditional_prob(node, class_logits):
    """
    Returns P(node | parent(node)).
    """
    parent = node.parent
    siblings = parent.children

    sibling_indices = [child.logit_index for child in siblings]
    sibling_logits = class_logits[sibling_indices]

    sibling_probs = F.softmax(sibling_logits, dim=0)

    node_position = siblings.index(node)
    return sibling_probs[node_position]

def path_from_root(node):
    path = []
    current = node

    while current.parent is not None:
        path.append(current)
        current = current.parent

    path.reverse()
    return path


def wordtree_probability(target_node, class_logits):
    """
    Computes P(target_node) as product of conditional probabilities.
    """
    prob = torch.tensor(1.0, device=class_logits.device)

    for node in path_from_root(target_node):
        prob = prob * conditional_prob(node, class_logits)

    return prob
```

YOLOv2 also uses Darknet-19 as the backbone. Darknet-19 is a 19-layer CNN.

## [YOLOv3](https://arxiv.org/pdf/1804.02767) (Joseph Redmon, Ali Farhadi, 2018)

[Code](https://pjreddie.com/darknet/yolo/)

- Multiscale prediction: bounding boxes are predicted at 3 scales, similar to a feature pyramid. The paper notes that YOLO historically struggled with small objects, and multiscale prediction improved this substantially.
- Better and bigger backbone: 53-layer Darknet-53, which uses residual connections and is more efficient than ResNet-101/152 in this context.
- Classifier: **allows a single box to have multiple labels** instead of softmax. Each class gets its own sigmoid probability: `class_probs = sigmoid(class_logits)`, so one box can have probabilities like `Person: 0.97, Woman: 0.91, Athlete: 0.74, Dog: 0.01`.
- Fixed a critical dataloading bug from YOLOv2, with about a 2% mAP improvement.

On anchors and grid cells:

```
# anchors across different grid sizes 
"13x13": [(116,90), (156,198), (373,326)],  
"26x26": [(30,61), (62,45), (59,119)],  
"52x52": [(10,13), (16,30), (33,23)]
```

During training and inference, for each scale/cell/anchor:

- the network predicts center offsets `tx, ty` and size adjustments `tw, th`;
- the cell location (for example, `(5, 7)`), anchor size (for example, `(156, 198)`), and stride are already known;
- the final decoded output is box center `bx, by`, size `bw, bh`, class scores, and objectness.

Training data here is primarily COCO.

## [YOLOv4](https://arxiv.org/pdf/2004.10934) (Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao, April 2020)

YOLOv4 is still a one-stage detector. A key improvement is its three-part architecture:

- Backbone: CSPDarknet53, using the [Cross Stage Partial (CSPNet)](https://arxiv.org/abs/1911.11929) strategy on top of Darknet-53.
- Neck: improved [Spatial Pyramid Pooling (SPP)](https://arxiv.org/abs/1406.4729) and PAN (Path Aggregation Network) for richer multi-scale feature aggregation.
- Head: YOLOv3-style anchor-based detection head.

![](https://i.postimg.cc/nzpxGc4q/Screenshot-from-2026-05-21-15-29-50.png)

### CSPNet (Cross Stage Partial)

The CSPNet paper argues that many CNN stages produce duplicate gradient information. Splitting and merging feature paths can reduce compute while improving gradient diversity. For example, a normal residual stage looks like:

```python
def normal_stage(x):
    x = block1(x)
    x = block2(x)
    x = block3(x)
    return x
```

If `x` has 256 channels, some gradients can be repetitive:

```
256 channels enter
256 channels get processed repeatedly
256 channels exit
```

Instead, CSP splits the input feature map into two groups. The first group goes through heavier blocks while the second group skips them. **Note that this step is deterministic**. Finally, the two groups are concatenated:

```python
def csp_stage(x):
    x1, x2 = split_channels(x)
    y1 = heavy_blocks(x1)
    y2 = x2
    # Merge both paths
    y = concat(y1, y2)
    # Mix information across channels
    y = transition_conv(y)

    return y
```

So CSP says not every channel needs expensive transformation. Some channels preserve original information while others are transformed.

### SPP (Spatial Pyramid Pooling)

YOLO detectors usually do not use the original 2014 SPPNet formulation. They use a simpler version:

```
feature map
   ├── max pool with small kernel
   ├── max pool with medium kernel
   ├── max pool with large kernel
   └── concatenate all results
```

Convolutional layers have limited receptive fields. SPP augments context by letting each location aggregate information from different neighborhood sizes. Because max pooling uses stride/padding settings chosen per kernel, outputs can keep the same `H x W` shape.

For example, with `kernel_sizes = [5, 9, 13]`, SPP generates:

```
x.shape = [B, 512, 13, 13]
p5  = maxpool(x, kernel_size=5,  stride=1, padding=2) # [B, 512, 13, 13]
p9  = maxpool(x, kernel_size=9,  stride=1, padding=4) # [B, 512, 13, 13]
p13 = maxpool(x, kernel_size=13, stride=1, padding=6) # [B, 512, 13, 13]
```

Then concatenate them:

```
y = concat([x, p5, p9, p13], dim="channels")
y.shape = [B, 2048, 13, 13]
```

Usually, a `1x1` convolution is applied to mix/compress the output tensor.

SPP is typically placed at the end of the backbone or the start of the neck. If placed too early, it aggregates mostly low-level texture patterns and is more expensive. Late backbone features are smaller and semantically richer, so SPP is both cheaper and more meaningful there.

## YOLOv5 (June 2020)

[Code](https://github.com/ultralytics/yolov5) (same Ultralytics ecosystem as YOLOv8 and YOLO11)

Architecturally, YOLOv5 is closer to the YOLOv3/YOLOv4 generation than to YOLOv8: it uses a PyTorch implementation with a CSP-style backbone/neck and anchor-based detection, while YOLOv8 later introduced an anchor-free split Ultralytics head. YOLOv5’s popularity came from its practical tooling: **PyTorch implementation, clean training pipeline, pretrained model zoo, test-time augmentation, model ensembling, and easy export to deployment formats such as ONNX, CoreML, TensorRT, and TFLite**

Training on your own data is an iterative loop: collect and label data, train, deploy, inspect failures/edge cases, add more examples, and retrain.

1. Collect images, label objects with bounding boxes
2. Organize the dataset into YOLO format, create a dataset YAML file listing train/validation
3. Image paths and class names, then run YOLOv5’s `train.py` using a pretrained checkpoint such as `yolov5s.pt`
4. After training, the best model is saved as something like `runs/train/exp/weights/best.pt`; you can validate it, run inference on new images, then export that trained checkpoint to a deployment format.

#### SPPF (Spatial Pyramid Pooling - Fast): An Approximation to SPP

Instead of

```
p1 = maxpool_5x5(x)
p2 = maxpool_9x9(x)
p3 = maxpool_13x13(x)
```

SPPF repeatedly applies a `5x5` max pool:

```
p1 = maxpool_5x5(x)  # 5x5 receptive field
p2 = maxpool_5x5(p1) # 9x9 receptive field
p3 = maxpool_5x5(p2) # 13x13 receptive field

out = concat([x, p1, p2, p3])
```

## [YOLOv6](https://arxiv.org/pdf/2209.02976) (Meituan Inc, Sep 2022)

[Code](https://github.com/meituan/YOLOv6/)

YOLOv6 is a practical real-time object detection system. It runs well on commonly available hardware such as NVIDIA Tesla T4 GPUs. For the backbone, it uses EfficientRep, a hardware-aware CNN design. Smaller models (YOLOv6-N/S) use RepBlock, while larger models (YOLOv6-M/L) use CSPStackRep blocks to improve representational capacity. For the neck, YOLOv6 adopts Rep-PAN, which builds on PAN-style feature aggregation with re-parameterized blocks. For the detection head, YOLOv6 introduces an Efficient Decoupled Head that separates classification and regression while simplifying convolutions to reduce cost.

### Hardware-aware CNN

"Hardware-aware" mainly means the network is designed around operations that are fast and easy to optimize during inference. For example, YOLOv6 uses RepBlock, inspired by **re-parameterization** ideas like RepVGG. During training, the block has multiple branches; during inference, those branches are mathematically fused into a simpler convolutional structure.

### RepBlock (Re-parameterized Block)

Convolution is linear, and batch normalization can be folded into convolution at inference. If we have `3x3`, `1x1`, and identity branches (plus BN), can we combine them? Yes.
Example:

For identity,

```
Identity_kernel =[[0, 0, 0], [0, 1, 0], [0, 0, 0]]
```

For `K1_fused = [[5]]`,

```
K1_as_3x3 =[[0, 0, 0], [0, 5, 0], [0, 0, 0]]
```

Then with kernel=3 convolution,

```
K3 =[[1, 2, 1], [0, 1, 0], [2, 1, 0]]
```

In total you would have a convolution equivalent to:

```
K_final = K3_fused + K1_as_3x3 + Kid_as_3x3
K_final =  
[[2, 4, 2],  
[0, 7.5, 0],  
[4, 2, 0]]
```

So during training, we keep the three branches separate. The identity branch is especially important because it preserves input signal and improves gradient flow, similar in spirit to residual connections. This makes optimization more stable, even though an equivalent fused `3x3` kernel can be used at inference.

```
branch A: 3x3 Conv + BN
branch B: 1x1 Conv + BN
branch C: Identity + BN
```

### PAN (Path Aggregation Network)

PAN is usually in the neck, where it fuses early high-resolution feature maps with later low-resolution feature maps.

```
Backbone outputs:

P3: high resolution, fine details      80×80
P4: medium resolution                  40×40
P5: low resolution, strong semantics   20×20


FPN top-down path:

P5 ──upsample──► P4
                 │
                 ▼
              fused P4 ──upsample──► P3
                                      │
                                      ▼
                                   fused P3


PAN bottom-up path:

fused P3 ──downsample──► fused P4
                         │
                         ▼
                      fused P4 ──downsample──► fused P5
                                               │
                                               ▼
                                            final P5
```

- Open question: what is the exact detection head architecture in this variant?

## [YOLOv7](https://arxiv.org/pdf/2207.02696) (Jul 2022, Institute of Information Science, Academia Sinica Taiwan)

[Code](https://github.com/WongKinYiu/yolov7)

YOLOv7 is not a radical reinvention of the YOLO architecture, but rather a set of highly practical, fine-grained improvements designed to make real-time object detection more accurate without sacrificing speed. Its main contribution is the use of **trainable Bag-of-Freebies**: techniques that improve training and representation learning while keeping inference costs low. These include planned **re-parameterized convolutions,** **coarse-to-fine lead-guided label assignment**, and extended compound scaling strategies for detection models. In other words, YOLOv7 improves the balance between accuracy and efficiency not by simply making the model larger, but by making the training process and architecture design smarter.

### Coarse-to-Fine Lead-Guided Label Assignment

In YOLOv7, an **auxiliary head is attached before the final lead-head path is fully completed**, usually from an intermediate feature stage. It provides **deep supervision**, giving earlier/shallow layers a stronger learning signal during training. The paper describes using lead-head predictions plus ground truth to generate **coarse-to-fine hierarchical labels**: coarse labels train the auxiliary head, and fine labels train the lead head.

Lead-guided label assignment is like a teacher-student strategy inside the detector.

$$
L_{total} = L_{lead}^{fine} + \lambda L_{aux}^{coarse}
$$

The lead head, which is responsible for the final prediction, is treated as the more reliable “teacher.” Its predictions are used together with the ground-truth boxes to generate soft labels. However, YOLOv7 does not give the same labels to every branch. The auxiliary head receives **coarser, more relaxed labels**, meaning more candidate boxes are treated as positives so the earlier layers get a stronger and easier learning signal. The lead head receives **finer, stricter labels**, focusing only on the best-matching candidates for precise final detection. In practice, this helps the model learn broadly first and refine later: the auxiliary branch improves recall and feature learning, while the lead branch sharpens precision. This is why the method is called **coarse-to-fine lead-guided label assignment**.

![](https://i.postimg.cc/VvT2vZTJ/aaccf667-518c-431e-bcaa-d5e99b9f08fa.png)

Suppose YOLOv7 has 6 candidate prediction boxes near that object. The lead head predicts the following matching scores, where higher means “this candidate looks like a good match to the ground truth.”

| Candidate | IoU with GT | Class confidence | Combined score |
| --------- | ----------: | ---------------: | -------------: |
| A         |        0.82 |             0.90 |           0.74 |
| B         |        0.76 |             0.85 |           0.65 |
| C         |        0.61 |             0.80 |           0.49 |
| D         |        0.48 |             0.75 |           0.36 |
| E         |        0.35 |             0.70 |           0.25 |
| F         |        0.20 |             0.60 |           0.12 |

A **fine** label assignment might say:

> Only candidates with score above **0.50** are positive.

So the **lead head** trains on:

|Positive for lead head?|Candidates|
|---|---|
|Yes|A, B|
|No|C, D, E, F|
A **coarse** label assignment might relax the threshold:

> Candidates with score above **0.30** are positive.

So the **auxiliary head** trains on:

|Head|Label style|Positive candidates|Purpose|
|---|---|---|---|
|Auxiliary head|Coarse|A, B, C, D|More recall, easier learning|
|Lead head|Fine|A, B|More precision, final detection quality|
