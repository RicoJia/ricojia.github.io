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