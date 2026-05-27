---
layout: post
title: YOLO V8 - 26 Models Review 
date: 2026-04-14 13:19
subtitle: ""
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine-Learning
---

## [YOLOv8](https://github.com/ultralytics/ultralytics) (Jan 10, 2023, Ultralytics)

 In spirit, YOLOv8 follows the same direction as YOLOv5: it emphasizes ease of use, fast deployment, and a strong balance between detection accuracy and inference speed.
 
Architecturally, YOLOv8 introduces an **improved backbone and neck**, together with an **anchor-free detection head**. The anchor-free design removes the need for manually predefined anchor boxes, which simplifies the detection pipeline and can improve flexibility across datasets. Beyond detection, Ultralytics also provides YOLOv8 variants for instance segmentation, classification, pose/keypoint detection, and oriented object detection, making it a broader computer vision framework rather than only a single object detector.

### Improved Backbone and Neck

YOLOv8 uses [**C2f modules** instead of older YOLOv5-style **C3 modules**](https://deepwiki.com/DataXujing/YOLOv8/2-yolov8-architecture). C2f roughly means **Cross Stage Partial with two convolutions / faster feature fusion**. It improves gradient flow and feature diversity through lightweight bottleneck paths and concatenation.

### Decoupled Anchor-Free Detection Heads

Problem: anchors are hand-designed or dataset-tuned. If your dataset has unusual object shapes, like long fish, tiny marine debris, or robot parts, anchor choices may not fit well.


YOLOv8 uses an **anchor-free detection head**, which means it does not rely on predefined anchor boxes. YOLOv8 instead treats each feature-map location as an implicit reference point in the original image. From this point, the model predicts the geometry of the bounding box, such as the distances to the left, top, right, and bottom edges, along with the object class. 

YOLOv8 also uses a **decoupled head**, where classification and box regression are handled by separate branches. The classification branch predicts what object is present, while the box regression branch predicts where the object is located. This design simplifies the detection pipeline, removes the need for manually tuned anchors, and helps YOLOv8 adapt more flexibly to objects with different shapes and scales.


![](https://i.postimg.cc/yYfQbgj1/537d317d-9e9d-414f-8c4d-f14e8d2dd61f.png)

### Training UI is very clean (yet to see if it's fast)

```python
from ultralytics import YOLO    
model = YOLO("yolov8s.pt")  
  
# 2. Fine-tune on your dataset  
results = model.train( 
	data="data.yaml",  
	epochs=100,  
	imgsz=640,  
	batch=16,  
	device=0  
)  
  
# 3. Validate  
metrics = model.val()  
  
# 4. Run inference with your trained model  
trained_model = YOLO("runs/detect/train/weights/best.pt")  
results = trained_model.predict("test_image.jpg", save=True)
```

Source: https://docs.ultralytics.com/models/yolov8, https://www.mdpi.com/2073-431X/15/2/74

|Model|COCO mAP50–95|CPU ONNX latency|A100 TensorRT latency|Params|FLOPs|
|---|--:|--:|--:|--:|--:|
|YOLOv8n|37.3|80.4 ms|0.99 ms|3.2M|8.7B|
|YOLOv8s|44.9|128.4 ms|1.20 ms|11.2M|28.6B|
|YOLOv8m|50.2|234.7 ms|1.83 ms|25.9M|78.9B|
|YOLOv8l|52.9|375.2 ms|2.39 ms|43.7M|165.2B|
|YOLOv8x|53.9|479.1 ms|3.53 ms|68.2M|257.8B|

![](https://dfimg.dfrobot.com/enshop/image/cache3/Blog/13844/6.jpg)


## [YOLOv9](https://arxiv.org/pdf/2402.13616) (Feb 2024)

[Code](https://github.com/WongKinYiu/yolov9)

Instead of only improving the backbone, neck, label assignment, or detection head, YOLOv9 focuses on a deeper training problem: information loss inside deep networks. As an image passes through many layers of feature extraction and transformation, some original input information is compressed or discarded, which can make gradients less reliable.

### PGI

YOLOv9 addresses this with Programmable Gradient Information, or PGI, a training strategy that uses an auxiliary reversible branch to preserve richer input information and generate more useful gradient signals. The key idea is that the main network should not only learn from the final detection loss, but also receive better-guided supervision during backpropagation.  

The auxiliary branch is _reversible in design_, so it is less likely to destroy information while producing auxiliary training signals.

![](https://i.postimg.cc/sXwD1ysp/Screenshot-from-2026-05-21-22-32-47.png)

Importantly, this auxiliary branch helps during training and can be removed during inference, so it improves learning without adding extra inference cost.


![](https://i.postimg.cc/0QSVBVQf/d1a06386-6bb4-44d2-8daf-c026734c7167.png)

Green arrows represent gradients; purple arrows represent the **auxiliary reversible branch and multi-level auxiliary information connections**.

At the end of training: 
```
L_aux → gradient from auxiliary branch
L_total = L_main + λ L_aux

# So finally,
∂L_total/∂W = ∂L_main/∂W + λ ∂L_aux/∂W
```
#### RevBlock

A **Rev Block**, or reversible block, is a neural-network block designed so its input can be recovered from its output. Split the input into two parts:

```python
# Input feature x is split into two parts along the channel dimension
# x = [x1, x2]
def rev_block_forward(x):
    x1, x2 = split_channels(x)
    # F and G are normal neural network sub-blocks
    # for example: Conv-BN-Activation-Conv
    y1 = x1 + F(x2)
    y2 = x2 + G(y1)
    y = concat_channels(y1, y2)
    return y

# This is NOT used directly in PGI; it demonstrates that x can be recovered from y.
def rev_block_reverse(y):
    y1, y2 = split_channels(y)
    # Reverse the second equation first:
    # y2 = x2 + G(y1)
    x2 = y2 - G(y1)
    # Then reverse the first equation:
    # y1 = x1 + F(x2)
    x1 = y1 - F(x2)
    x = concat_channels(x1, x2)
    return x
```

In PGI, a **RevBlock** is not just another convolution block. It is used in the auxiliary training branch to keep information available for gradient computation. A useful mental model is:

```
# Main branch: used at inference
main_features = main_backbone(image)
main_pred = detection_head(main_features)
L_main = detection_loss(main_pred, targets)

# Auxiliary reversible branch: training only
aux_features = rev_branch_forward(image_or_early_features)
aux_pred = auxiliary_head(aux_features)
L_aux = detection_loss(aux_pred, targets)

# Combined training objective
L_total = L_main + lambda_aux * L_aux
L_total.backward()
optimizer.step()

# During inference:
# use only main_backbone + detection_head
# remove rev_branch + auxiliary_head
```

### GELAN

Alongside PGI, YOLOv9 introduces GELAN, or Generalized Efficient Layer Aggregation Network. GELAN can be understood as a more flexible version of ELAN: it is designed to improve information flow through the network while remaining lightweight and efficient. Its structure is based on gradient path planning, meaning the architecture is built with the backward learning signal in mind, not just the forward prediction path. 


GELAN keeps part of the input path because not every feature should be forced through deep computation.

```python
def forward(self, x):
    # 1. Project input channels
    x = self.proj(x)  # Conv2d, usually 1x1

    # 2. Split into two channel groups
    x1, x2 = split_channels(x, num_splits=2)

    # 3. Keep short paths and create deeper paths
    y1 = x1                  # short path
    y2 = x2                  # short path

    y3 = self.path1_blocks(x2)
    y4 = self.path2_blocks(y3)

    # 4. Aggregate features from different depths
    y = concat_channels([y1, y2, y3, y4])

    # 5. Fuse aggregated channels
    out = self.fuse(y)       # Conv2d, usually 1x1

    return out
```

Note: residual connection is a simpler version of GELAN. Residual connection merges by **addition**, GELAN/ELAN merges by **concatenation then fusion**.

```python
def forward(x):
    return x + Block(x)
```

Together, PGI and GELAN make YOLOv9 stand out because they shift attention from only **“how do we predict better boxes?” to “how do we preserve useful information and provide better gradients while training?”** 

![](https://i.postimg.cc/cJTNBmjb/Screenshot-from-2026-05-21-22-34-54.png)

YOLOv9 emphasizes information flow and gradient quality. The introduction of PGI and GELAN differentiates it from earlier versions by improving training dynamics, not just inference-side architecture.

## [YOLOv10](https://arxiv.org/pdf/2405.14458) (May 2024)

[Code](https://github.com/THU-MIG/yolov10)

A major limitation of many earlier YOLO models is **Non-Maximum Suppression (NMS)** as a post-processing step to remove duplicate predictions. Although effective, NMS adds extra latency and makes deployment less clean because the model is not fully end-to-end. 

- NMS can become a latency bottleneck: when the scene is crowded and the number of candidate boxes increases, post-processing time can grow and inference latency becomes less predictable.
- NMS is sensitive to manually chosen hyperparameters, especially the IoU threshold. If the threshold is too aggressive, nearby or overlapping objects may be incorrectly removed; if it is too loose, duplicate boxes may remain. 

YOLOv10 addresses this with an **NMS-free detection strategy** based on **consistent dual assignments**. During training, it uses both one-to-many assignment, which provides rich supervision, and one-to-one assignment, which encourages each object to be matched by a single high-quality prediction. At inference time, YOLOv10 keeps the one-to-one head, allowing it to produce final detections without NMS. 

In addition, YOLOv10 introduces a holistic efficiency-accuracy driven design, including lightweight classification heads, spatial-channel decoupled downsampling, and rank-guided block design, to reduce computational cost while preserving detection accuracy. Together, these changes improve the latency-accuracy tradeoff and make YOLOv10 better suited for real-time deployment.

![](https://pica.zhimg.com/v2-bc6514429987175482232ea4fdc0aabc_1440w.jpg)

### One-to-many head

The **one-to-many head** is similar to standard YOLO training. For each ground-truth object, many nearby feature-map locations can be treated as positives.

```
One object:
    fish ground truth
        ↓
    many positive prediction locations
        ↓
    rich training signal
```

This is good for learning because the model gets many gradients from one object. It helps the backbone and neck learn useful features. But at inference, this creates a problem:

```
same fish → many predicted boxes
```

So traditional YOLO needs **NMS** to remove duplicates.

### One-to-one head

The **one-to-one head** is stricter. For each ground-truth object, only one prediction is selected as the positive match.

```
One object:
    fish ground truth
        ↓
    one best prediction location
        ↓
    one final box
```
This is what makes NMS-free inference possible. If the model learns to assign only one strong prediction to each object, then you do not need NMS to clean up duplicate boxes.

But one-to-one training alone can be weaker because for one object, only one location gets trained as positive. That means fewer positive gradients.

So YOLOv10 combines both during training, and discards the one-to-many head during inference.

```python
def train_step(image, ground_truths):
    # Shared feature extractor
    features = backbone(image)
    features = neck(features)

    # Two heads during training
    preds_many = one_to_many_head(features)
    preds_one  = one_to_one_head(features)

    # Assign labels differently
    targets_many = assign_one_to_many(preds_many, ground_truths)
    targets_one  = assign_one_to_one(preds_one, ground_truths)

    # Compute losses for both heads
    loss_many = detection_loss(preds_many, targets_many)
    loss_one  = detection_loss(preds_one, targets_one)

    # Joint training
    loss = loss_many + loss_one

    loss.backward()
    optimizer.step()

    return loss
    
def inference(image):
    features = backbone(image)
    features = neck(features)
    # Use only the one-to-one head
    preds = one_to_one_head(features)
    # No NMS needed
    detections = decode_predictions(preds)
    return detections
```

### Matching Score

YOLOv10 uses the idea of a **consistent matching metric** so the one-to-many and one-to-one heads are not learning contradictory assignments. The metric balances classification confidence and localization quality, so a good match should both classify the object correctly and overlap the ground-truth box well. A simplified pseudo example is 

```python
def matching_score(pred, gt, alpha=1.0, beta=6.0):
    cls_score = pred.class_prob[gt.class_id]
    iou_score = iou(pred.box, gt.box)

    return (cls_score ** alpha) * (iou_score ** beta)
```

![](https://pica.zhimg.com/v2-6832eaba918ecb41b7615d66da3538d2_1440w.jpg)

## YOLO11 (Sep 10, 2024, Ultralytics)

[Code](https://github.com/ultralytics/ultralytics)

YOLO11 made feature extraction more efficient while preserving or slightly improving accuracy. One important change is the introduction of the C3k2 block, an efficient CSP-style module that replaces many of the earlier C2f blocks used in YOLOv8-style designs. Conceptually, C3k2 keeps the benefits of partial feature reuse and gradient flow, but organizes the computation more efficiently so the model can achieve strong performance with fewer parameters and lower computational cost. 

YOLO11 also introduces the C2PSA attention block after the SPPF stage, adding position-sensitive/spatial attention to help the model focus on more informative regions of the feature map. Together, C3k2 and C2PSA make YOLO11 feel less like a revolutionary redesign and more like a careful architectural refinement: the model becomes lighter and more efficient while maintaining the real-time detection behavior that makes the YOLO family useful.

### C3K2

**C3k2** is a **lighter CSP/C2f-style feature block**. Its job is: 

```
Take input features
↓
split them into shortcut + processed paths
↓
process only part of the channels through small bottleneck blocks
↓
concatenate the paths
↓
fuse everything with a final Conv2d
```

So conceptually, it is not a totally new idea. The real difference is that a C3k block is a mini CSP module that contains bottlenecks inside it.

```python
class C3k:
    def forward(self, x):
        a = Conv1x1(x)  # processed path
        b = Conv1x1(x)  # short CSP path

        a = bottleneck(a)
        a = bottleneck(a)

        return Conv1x1(concat([a, b]))


# Whereas:
class Bottleneck:
    def forward(self, x):
        return x + Conv(Conv(x))
```

### C2PSA

Same deal with the conventional cross-stage-partial pipeline 

```
input
  ↓
1×1 Conv projection
  ↓
split channels into two parts
  ├── short path: keep one part mostly unchanged
  └── attention path: send other part through PSA blocks
  ↓
concat short path + attention path
  ↓
1×1 Conv fuse
```

In principle, 
```python
class C2PSA:
    def forward(self, x):
        # 1. Project channels
        x = Conv2d(x)
        # 2. Split into two feature groups
        x_short, x_attn = split_channels(x, num_splits=2)
        # 3. Keep one branch short
        y_short = x_short
        # 4. Apply position/spatial attention to the other branch
        y_attn = x_attn
        for block in self.psa_blocks:
            y_attn = block(y_attn)
        # 5. Concatenate short branch + attention branch
        y = concat_channels([y_short, y_attn])
        # 6. Fuse channels
        out = self.cv2(y)
        return out
```

So the real difference is in the PSA block - **it uses an attention**:

```python
class PSABlock:
    def __init__(self, channels):
        self.attn = Attention(channels)
        self.ffn = FeedForward(channels)
    def forward(self, x):
        # Attention branch
        x = x + self.attn(x)
        # Feed-forward / conv branch
        x = x + self.ffn(x)
        return x
```

---
## YOLO26 (September 2025, Ultralytics)

[YOLO26: An Analysis of NMS-Free End to End Framework for Real-Time Object Detection](https://arxiv.org/abs/2601.12882) 

Again, **Non-Maximum Suppression (NMS)** remains a practical pain point. YOLOv10 pioneered an NMS-free direction for YOLO, and YOLO26 further pushes this idea with a native end-to-end design that produces final predictions directly without a separate NMS stage.

### 1. One-to-One Prediction Head

During training, the model is “taught” to predict only one most accurate box for each real object, so the model’s output is already the final result, requiring no additional filtering — clean and straightforward. This turns the inference process **into a deterministic mapping from input to output**, allowing latency to remain stable no matter how complex the scene is.

### 2. DFL-Free Head Designed for Edge Deployment

Recent YOLO versions, such as YOLOv8, commonly use **Distribution Focal Loss (DFL)** to pursue higher accuracy. DFL treats coordinates as a probability distribution. Although this improves localization accuracy, the **Softmax operation it relies on can be very time-consuming on edge devices such as NPUs and DSPs**, and **it is difficult to quantize**. This creates a significant **“export gap”**: the model may run extremely fast on a GPU, but perform poorly once deployed on edge hardware.
  

![](https://pic2.zhimg.com/v2-5b9f0ee6fa5bd10a90e743341d1826ed_1440w.jpg)


YOLO26 decisively abandons DFL, as shown on the right side of the figure above, and returns to simpler, more hardware-friendly **direct coordinate regression**. This “rollback” may look like a compromise in accuracy, but in reality, it reflects a strong focus on practical deployment performance.

This raises an important question: after removing two powerful tools, NMS and DFL, how can the model maintain accuracy and still train stably? YOLO26 addresses this with three major training techniques.

1. **MuSGD optimizer** ([blog post](https://ricojia.github.io/2026/04/19/muon-sgd/))
2. **[STAL](https://ricojia.github.io/2026/04/16/stal/)**
3. **[ProgLoss](https://ricojia.github.io/2026/04/16/progloss/)**


YOLO26’s excellent architecture also makes it a unified multi-task framework. Beyond object detection, it natively supports instance segmentation, pose estimation, oriented bounding box detection, and even open-vocabulary detection through YOLOE-26, demonstrating its significant potential as a next-generation visual foundation model.

What struck me most about YOLO26 is that the field of object detection is shifting away from an arms race focused solely on benchmark metrics and toward deeper thinking about real-world deployment value. Unlike many studies that stack increasingly complex modules to push mAP higher, YOLO26 boldly takes a “less is more” approach. By removing NMS and DFL, it directly addresses the long-standing “Export Gap” problem that has troubled the industry for years.

This shift may be more meaningful than performance numbers alone. After all, the pursuit of accuracy should be grounded in practical usability.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://pic3.zhimg.com/v2-0f47d04af45ee4dd270dae1b2900999e_1440w.jpg" height="300" alt=""/>
    </figure>
</p>
</div>

## Summary

![](https://pic1.zhimg.com/v2-935cbb6b827311a9a95057f50cbbf0dc_1440w.jpg)

Mental model focus:
- **NMS-free inference**
- **DFL removal**
- **ProgLoss**
- **STAL**
- **MuSGD**

Rather than following a DETR-style transformer redesign, these models emphasize deployment-friendly training objectives and decoding behavior.

YOLO families also use attention-like modules (for example, PSA), especially in later stages, but these are targeted feature-fusion/efficiency components, not full transformer encoder-decoder detection pipelines.

Overall, this points to a fast front end for 2D detection, OBB, and segmentation tasks.

## What YOLO Does in an Underwater Object Detection Dataset

For example, in an underwater robot dataset:

```text
small object: distant fish / small tool
medium object: nearby marine animal
large object: diver / robot arm close to camera
```

A deep feature map has strong semantic understanding but poor spatial detail. A shallow feature map has good spatial detail but weaker semantic understanding.

The neck combines them:

```text
shallow feature -> where exactly is the object?
deep feature    -> what is the object?
neck fusion     -> detect it accurately
```

This is especially important for small objects because small objects can disappear in deep layers after repeated downsampling.

## Underwater Object Detection

- [PLOS One underwater detection paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0311173)
- <https://arxiv.org/html/2506.23505v1>
- 2024: <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10752531>

## Datasets

- [URPC 2020](https://www.kaggle.com/datasets/lywang777/urpc2020?resource=download) (Underwater Robot Professional Contest, China): 5543 images with four object types: holothurian, echinus, scallop, and starfish.

- [DUO dataset](https://figshare.com/articles/dataset/DUO_zip/25370527?file=45670494)
- [Shared dataset drive](https://drive.google.com/drive/folders/1Fv14CpnAjmN642m7lFjHL8yRfeyHYFCQ)
- [Deep Fish](https://www.kaggle.com/datasets/vencerlanz09/deep-fish-object-detection)
- [Fish4Knowledge](https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/resources.htm)
