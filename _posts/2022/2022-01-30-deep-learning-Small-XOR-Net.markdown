---
layout: post
title: Deep Learning - Start Easy, Things I Learned From Training Small Neural Nets 
date: '2022-01-28 13:19'
subtitle: Basic Torch Network With Some Notes on Syntax
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Introduction

To gain some insights into how hyper parameters impacts training, I created a simple neural network using PyTorch to learn 2D input data. Specifically, I'm interested in exploring the impacts of:

- Weight Initialization
- Optimizer choice (SGD, momentum, RMSProp, Adam)

On:

- Gradient norms across layers
- Final cost  

Along with that, I created a visualization suite which could be used to visualize higher dimension Fully Connected neural nets as well. [For full code, please check here](https://github.com/RicoJia/Machine_Learning/blob/master/RicoNeuralNetPrototype/utils/debug_tools.py)

## Simple Neural Network

The neural net model looks like:

```python
import torch.nn as nn
class SimpleNN(nn.Module):
    def __init__(self, fcn_layers, initialization_func = nn.init.xavier_uniform_):
        super(SimpleNN, self).__init__()
        self.fcn_layers = fcn_layers
        [initialization_func(l.weight) for l in self.fcn_layers if isinstance(l, nn.Linear)]
        
        # Torch requires each layer to have a name
        for i, l in enumerate(self.fcn_layers):
            setattr(self, f"fcn_layer_{i}", l)
    
    def forward(self, x):
        for l in self.fcn_layers:
            x = l(x)
        return x
```

A simplified version of the driver code:

```python
import torch
import torch.nn as nn
import torch.optim as optim
def test_with_model(X_train, y_train, X_test, y_test, X_validation=None, y_validation=None):
    model = SimpleNN(
        fcn_layers=[
            nn.Linear(2, 4), 
            nn.ReLU(), 
            nn.Linear(4, 1), 
            nn.Sigmoid()
        ]
    )

    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    epochs = 1000
    for epoch in range(epochs):
        mini_batches = create_mini_batches(X_train, y_train, batch_size=64) 
        for X_train_mini_batch, y_train_mini_batch in mini_batches:
            optimizer.zero_grad()  # Zero the gradient buffers
            X_train_mini_batch = torch.Tensor([[0,0],[0,1], [1,0], [1,1]])
            y_train_mini_batch = torch.Tensor([0,1,1,0]).view(-1,1)
            # Forward pass
            outputs = model(X_train_mini_batch)
            loss = loss_func(outputs, y_train_mini_batch)  
            # Compute gradients using back propagation. Specifically, autodiff and computational graph is used here
            loss.backward()
            debugger.record_and_calculate_backward_pass(loss=loss)
            # Parameter update with gradients. Momentum, RMSProp are applied here.
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            with torch.no_grad():
                output = model(X_validation)
                loss = loss_func(output, y_validation)
                print(f'Validation Loss: {loss}')
```


## Experiements

### SGD Optimizer

- 💡 In a typical successful run, the weight and bias norms initially increase across all layers, then decrease to almost 0 and oscillates around there (so learning is stablized)


<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/cbb2bb84-f20b-45fb-9093-5303ea8cf578" height="300" alt=""/>
    </figure>
</p>
</div>

- 💡 Biases are usually initialized to 0, so it's trivial for analysis. Weights however, needs to be initialized carefully. For `ReLU` activation functions, we use `He` Initialization. Here we are using `sigmoid`, so we use `Xavier` initialization. Xavier/Glorot randomly initializes weights to 0 mean, $gain * \sqrt{\frac{6}{n_{i}+n_{i+1}}}$ variance.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/0d31c36f-6322-4675-a63f-39ebf6c910f9" height="300" alt=""/>
    </figure>
</p>
</div>

- 💡 Initialization does create a difference. In some runs, gradients could be zeros, or they could stay high. So, early stopping is necessary!

### Hyper Parameter Tuning

In a Gaussian Mixture example, I have 5 mixtures of classes. The first architecture, with only 2 layers, could learn only up to <80% on the test set. Once I added another hidden layer, the non-linearity increases and the accuracy could hit >90%. Note that cost still looks a little noisy at the end, with gradient norm oscillating in $[0, 0.15]$ in some cases. **However, since the eventual test set accuracy is decent, we don't need to worry too much about it**

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/84774ac6-a83b-457e-8ca2-fdb81d8956af" height="300" alt=""/>
    </figure>
</p>
</div>

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/dee6ef7c-185d-4ea0-8c71-3a1ee8692ec2" height="300" alt=""/>
    </figure>
</p>
</div>

Number of Epochs could matter, too. In a "circle within circles" examle, at first I tried a larger epoch number. The accuracy improves:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/8f99e367-99ee-44aa-af9c-816d4aa6bb7b" height="600" alt=""/>
    </figure>
</p>
</div>

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/5dd5be64-7f9a-4158-ae0d-dc8ed576742e" height="300" alt=""/>
    </figure>
</p>
</div>


### Adam Optimizer

In this example, the Adam optimizer does have higher convergence speed.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/4d9600b3-4340-4e9a-8d3f-247a2be64697" height="300" alt=""/>
    </figure>
</p>
</div>

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/dd603246-1409-4ece-a46d-cdfe2da810b0" height="300" alt=""/>
    </figure>
</p>
</div>

## Final Thoughts

For high productivity, it would be nice to build a training pipeline with enough parallel compute such that:

- The pipeline should be able to save weights, and statistics, and ideally, debugging visualization for futher analysis
- It's equipped with an early stopping mechanism which detects plateaus in test set validation. Once it has detected such a plateau, the pipeline starts a new network with the same hyper parameters but different initial parameters
- The pipeline is able to handle different combo of hyper parameters. I'd follow below sequence:
    1. Learning rate
    2. Model parameters: the number of layers, and activation functions.
    3. Optimizer choices: Adam vs SGD vs momentum only vs RMSProps only
