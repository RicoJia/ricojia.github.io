---
layout: post
title: Deep Learning - Hands-On ResNet Transfer Learning For CIFAR-10 Dataset
date: '2022-02-07 13:19'
subtitle: Data Normalization, Conv Net Training
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
    - Hands-On
---


## ResNet-50 Transfer Learning

**COMPLETE CODE can [be found here](https://github.com/RicoJia/Machine_Learning/blob/master/models/resnet-50-pretrained.ipynb)**

### Data Loading

Please see [this blogpost for data loading](2022-02-21-deep-learning-pytorch-data-loading.markdown)

## Model Definition

### PyTorch Built-In Model

```python
model = models.resnet50(weights='IMAGENET1K_V1') # This is close to the training result in paper. V2 is better
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
model = model.to(device)
```

This model however, might not be directly downloadable for [SSL protocol mismatch](https://ricojia.github.io/2024/08/18/rgbd-slam-setup-nvidia-orin-nano/#ssl-pitfall)

### Hand-Crafted RESNET-20

- `3x3` and `1x1` Conv layers for future uses

```python
def conv_3x3(in_channels, out_channels, stride, padding):
    return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )
    
def conv_1x1(in_channels, out_channels, stride):
    return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
        )
```

- Identity Block. Convolutional block (or projection block) is not used for RESNET-20's relatively small size

```python
class BasicIdentityBlock(nn.Module):
    """This is not the bottleneck block, it's the basic identity block
    Basic means there are 2 convolutions (3x3) back to back
    Identity means the skip connection does not require 1x1 convolution for reshaping 
    """
    def __init__(self, in_channels, out_channels, stride) -> None:
        # first conv layer is in charge of the actual stride
        super().__init__()
        self.conv1 = conv_3x3(in_channels=in_channels, 
                              out_channels=out_channels, stride=stride, padding=1,)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # second conv does same convolution
        self.conv2 = conv_3x3(in_channels=out_channels, 
                              out_channels=out_channels, stride=1, padding=1,)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        # since this is identity, we can add outputs with inputs together, if stride is 1
        # In case in_channels!=out_channels
        if stride != 1 or in_channels!=out_channels:
            # this is downsampling. Downsampling in ResNet is done thru conv layer 
            self.short_cut = nn.Sequential(
                conv_1x1(in_channels=in_channels, out_channels=out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.short_cut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        short_cut = self.short_cut(x)
        out += short_cut
        out = self.relu(out)
        return out
```


- RESNET-20 For CIFAR-10 dataset

```python

class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes) -> None:
        # input_shape = (64, 64, 3)
        super().__init__()
        # same padding, output 32x32x16
        output_channels = [16, 16, 32, 64]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=output_channels[0], kernel_size=(3, 3), stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=output_channels[0])
        # 32x32x16
        self.layer0 = self._make_layer(block, num_blocks[0], in_channels = output_channels[0], out_channels=output_channels[1], stride=1)  
        # 16x16x32
        self.layer1 = self._make_layer(block, num_blocks[1], in_channels = output_channels[1], out_channels=output_channels[2], stride=2)  # 16x16
        # 8x8x64
        self.layer2 = self._make_layer(block, num_blocks[2], in_channels = output_channels[2], out_channels=output_channels[3], stride=2)  # 8x8
        self.relu=nn.ReLU(inplace=True)
        # output 1x1x64
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=output_channels[3], out_features=num_classes)   #10

        self._initialize_weights()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_block, in_channels, out_channels, stride):
        # first block may downsample
        layers = [
            block(in_channels, out_channels=out_channels, stride=stride)
        ]
        for _ in range(num_block-1):
            layers.append(block(in_channels=out_channels, out_channels=out_channels, stride=1))
        return nn.Sequential(*layers)
            
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# resnet-20
model = ResNetCIFAR(block=BasicIdentityBlock, num_blocks=[3,3,3], num_classes=len(train_data.classes))
input_tensor_test = torch.randn(1,3,32,32)
output = model(input_tensor_test)
```

    - Note: in `self.relu=nn.ReLU(inplace=True)`, we use `inplace operation` which consumes no extra memory. So it's more friendly for large models.
    - `nn.Sequential(*layers)` is a container that allows stacking of layers. The number of layers is determined during **runtime**. During forward pass, input x is fed through the layers in sequence. During backward pass, back prop is conducted in sequence as well. (this is the very definition of a "Sequential model")

    - Batch norm layers `self.bn1` and `self.bn2` can't be shared because they have well, 4 different params each. (mean, variance, exponential decay's parameters )
    - `nn.Identity()` is basically no-op.
    - `m = nn.AdaptiveAvgPool2d((5, 7))`: given input `m x n x c`, output `5x7xc`.
    - `x = torch.flatten(x, start_dim=1)` flattens `[batch_size, 64, 1, 1]` to `[batch_size, 64]`. 

### Model Training

When training, I'm not sure why a batch of 64 images would crash Nvidia Orin (7.4G GPU Memory). A batch of 16 images is fine. But, **I've observed that a simulated batch size of 64 or 256** yields a training accuracy of 91%, and each batch takes 160s. However, a batch of 16 **caps at a training accuracy of 71% and each batch takes 1600s** on an Nvidia Orin Nano. Batch simulation is to run backward propagation after `N` batches.

Another observation is the use of `lr_scheduler`. It truly helped reduce gradient oscillation when training accuracy caps.

```python
import time
# Define the training function
MODEL_PATH = 'resnet_cifar10.pth'
ACCUMULATION_STEPS = 8
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        start = time.time()
        print(f'Epoch [{epoch + 1}/{num_epochs}] ')
        
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            # This is because torch.nn.CrossEntropyLoss(reduction='mean') is true, so to simulate a larger batch, we need to further divide
            loss = criterion(outputs, labels)/ACCUMULATION_STEPS
            # Backward pass and optimization
            loss.backward()
        
            if (i+1)%ACCUMULATION_STEPS == 0:
                optimizer.step()
                # Zero the parameter gradients
                optimizer.zero_grad()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        # adjust after every epoch
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct_train / total_train
        print("correct train: ", correct_train, " total train: ", total_train)

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"epoch: {epoch}, saved the model. "
              f'Train Loss: {epoch_loss:.4f} '
              f'Train Acc: {epoch_acc:.2f}% ')

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=False, map_location=device))
    print("loaded model")
model.to(device)

criterion = nn.CrossEntropyLoss()
weight_decay = 0.0001
momentum=0.9
learning_rate=0.1
num_epochs=50
batch_size=16
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # 0.001 as learning rate is common
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

# Learning rate scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
```

Some other notes:

- `model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))`: torch models actually could have tensors for GPUs. So if your model is trained on a GPU, it can't be loaded onto a CPU. This can be mitigated by `model.load_state_dict(torch.load(MODEL_PATH, map_location=device))`
- `loss.item()` gives the average loss across the current batch
- `model.eval()` vs `model.train()`: in the 'eval' mode, dropout and batch normalization are turned off
-  0.001 as learning rate is common for `Adam`, 0,1 is common for SGD.

### Result

Evaluation Code

```python
    # Evaluation phase
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        # for inputs_test, labels_test in train_loader:
        
        # TODO I AM ITERATING OVER TRAIN_LOADER, SO I'M MORE SURE
        for inputs_test, labels_test in test_loader:
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)
            outputs_test = model(inputs_test)
            _, predicted_test = outputs_test.max(1)
            total_test += labels_test.size(0)
            correct_test += predicted_test.eq(labels_test).sum().item()

    test_acc = 100. * correct_test / total_test

    # Adjust learning rate
    end = time.time()
    print("elapsed: ", end-start)
    print(f'Test Acc: {test_acc:.2f}%')

    print('Training complete')
```

Our final result is 61%, on the validation dataset of PASCAL VOC for these classes: `{'aeroplane', 'car', 'bird', 'cat', 'dog', 'frog'}`
