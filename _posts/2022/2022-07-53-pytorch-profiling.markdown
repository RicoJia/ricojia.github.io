---
layout: post
title: Deep Learning - PyTorch Profiling
date: '2022-07-25 13:19'
subtitle: PyTorch Profiler
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Sample Code

```python
#!/usr/bin/env python3
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

def memory_consumption():
    print("Memory Consumption")
    model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)
    with profile(activities=[ProfilerActivity.CPU],
            profile_memory=True, record_shapes=True) as prof:
        model(inputs)
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

def execution_time_profile():
    print("Memory Execution")
    model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)
    print("Help: operators can call other operators. self cpu time only o")

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))

if __name__ == "__main__":
    execution_time_profile()
    memory_consumption()
```

This code will yield a result like:

```
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                      aten::empty         0.37%     139.931us         0.37%     139.931us       0.700us      92.93 Mb      92.93 Mb           200  
    aten::max_pool2d_with_indices         9.85%       3.714ms         9.85%       3.714ms       3.714ms      11.48 Mb      11.48 Mb             1  
                 aten::empty_like         0.10%      38.093us         0.14%      53.831us       2.692us      47.37 Mb       1.91 Mb            20  
                      aten::addmm         0.56%     211.046us         0.60%     225.042us     225.042us      19.53 Kb      19.53 Kb             1  
                       aten::mean         0.08%      30.837us         0.27%     101.710us     101.710us      10.00 Kb       9.99 Kb             1  
                       aten::div_         0.05%      17.173us         0.11%      42.330us      42.330us           8 b           4 b             1  
              aten::empty_strided         0.01%       3.095us         0.01%       3.095us       3.095us           4 b           4 b             1  
                     aten::conv2d         0.15%      56.563us        76.36%      28.789ms       1.439ms      47.37 Mb           0 b            20  
                aten::convolution         0.41%     154.381us        76.21%      28.732ms       1.437ms      47.37 Mb           0 b            20  
               aten::_convolution         0.29%     110.645us        75.80%      28.578ms       1.429ms      47.37 Mb           0 b            20  
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
```

- [Aten](https://pytorch.org/cppdocs/#aten) is a tensor library that PyTorch's Python and C++ libs are built on. It has hundreds of operations, such as addition, mean, convolution, etc. with both the CPU and the GPU implementations
  - `aten::empty`: This is the ATen operation that creates an empty tensor with specified dimensions and data type.
- `Self CPU` time refers to the time spent exclusively on an operation. An operation can call other operations.
- `CPU Mem`: Total amount of CPU memory allocated across all calls, typically measured in MB. `Self CPU Mem` refers to CPU Mem exclusively consumed by the call specifically
- `# of Calls`: The number of times this operation was invoked during profiling.
