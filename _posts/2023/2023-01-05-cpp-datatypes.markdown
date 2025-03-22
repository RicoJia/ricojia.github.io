---
layout: post
title: C++ - Datatypes
date: '2023-01-05 13:19'
subtitle: std::optional, structural binding
header-img: "img/post-bg-alitrip.jpg"
comments: true
tags:
    - C++
---

## POD Types

- char or unsigned char is 1 byte
- float, int, are usually 4 bytes (32 bits) on a 64 bit machine
- double is 8 bytes (64 bits)
- long double is 8, or even 16 bytes

### Common Bugs

- Observed Error:  error: no matching function for call to ‘min(float, double)’

```cpp
// Ensure both arguments are of type float
rk_deltas_[i] = std::min(my_var, 2.0f);
```

## Optional

`std::optional` can specify if a value is valid or not. It has:

- `.has_value()` to examine if it has value
- `.value()` to retrieve the value

```cpp
#include <optional>
#include <iostream>
#include <vector>
using namespace std;

void test_optional(){
    auto get_ll = [](bool does_return)-> std::optional<size_t> {
        if(does_return) 
            return 1;
        else 
            return {};
    };
    cout<<get_ll(true).value()<<endl;
    if (get_ll(false).has_value()){
        cout<<"yay"<<endl;
    }
}

int main(){
    test_optional();
}
```

## Structural Binding
TODO