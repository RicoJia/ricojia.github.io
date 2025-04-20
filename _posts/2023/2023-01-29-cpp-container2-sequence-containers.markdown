---
layout: post
title: C++ - [Container 2] Sequence Containers
date: '2023-01-29 13:19'
subtitle: std::iota, std::generate, std::vector, std::array, std::list, std::deque, std::forward_list
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Introduction

Sequence containers in C++ include: `vector`, `array`, `list`, `forward_list`, `deque`. The common operations I look are are:

- Initialization
- Push front, push back
- Sequential Access, random access
- Element removal

Slightly more advanced operations are:

- Appending one container to another
- Resize, reserve

In the meantime, we want to look at the time complexity of each push and removal operations. Each type of container has their unque features as well. We will see them in the following sections.

## Vector Operations

```cpp
vector1.resize(vector1.size() + vector2.size());
// - Append `vector2` to the end of `vector1`
vector1.insert(vector1.end(), vector2.begin(), vector2.end());
```

- Appending an element to the end of vector can be done by `insert(target_it, source_end_it, source_end_it);`. 
- But once `vec.capacity() < vector1.size() + vector2.size()`, memory **reallocation** will happen. So the worst case is O(N).

### Memory Reallocation Of Vector

- vector under the hood is like an array, which uses contiguous memory. Therefore when no contiguous memory is found, **the whole vector needs to be moved over**
    - If the data is POD (Plain Old Data) like int, the data will be copied over. 
    - If the data has a move ctor, it will be moved over, otherwise copied. 

## List Operations

```cpp
// 1. list initialization
std::list<int> ls {1,2,3}; 
// 2. forward insert, backward insert
ls.push_front(0);
ls.push_back(1);
// 3. Access - No random access, just bi-directional access
for (const auto& m: ls){
    cout<<m<<", ";
}
cout<<endl;

ls.push_front(2);
ls.push_front(3);

// 4. remove 1: erase-remove idiom; O(1)
ls.erase(ls.begin());   // C++ 11
// 4. remove 2: std::erase()
std::erase(ls, 2);      // C++ 20    
// 4. remove 3 all occurences of 1, so it's o(n)    
ls.remove(1);

for (auto& m: ls){
    cout<<m<<endl;    // sew 1
}
```

Also, there are:

- `list.size()`
- `list.front()` and `list.back()`

Advanced Operations:

```cpp
list<int> ls{1,2,3};
list<int> ls2{1,2,3};
// 1. Appending a whole list
ls.splice(ls.end(), ls2);
    
// 2. Appending a single element: 
auto it = ls.begin();
std::advance(it, 1);
ls.splice(ls.end(), ls2, it);  

std::advance(it, 1);    // std::advance() is a void function
// 3. Appening a segment of list:
ls.splice(ls.end(), ls2, ls.begin(), it);
```

- appending one container to another: use `splice`. Splice means "connecting two ropes together through intertwining".
    - source `list ls2`  is always required in `std::list::splice`. See the above for 3 its 3 usages
    - This is `O(1)`, because the splice will just manipulate the iterator inside.

## Vector Operations

### Filling Operations

- Fill a vector with `0, 1,2,3,...`: `std::iota`. `iota` is the greek word for "small amount"

```cpp
#include <numeric>
std::vector<int> vec(10);
std::iota(vec.begin(), vec.end(), 0);
```

- Fill a vector with values with a custom lambda:

```cpp
#include <algorithm>
std::vector<int> vec(10,2); // now all 2
int x = 1;
std::generate(vec.begin(), vec.end(), [&x](){return x*=3;}); // 3, 9, ...
```
    - for `std::generate()`, the lambda must be a callable without args

- Fill the first `n` element with values:

```cpp
#include <algorithm>
std::vector<int> vec(10,2); // now all 2
int x = 1;
int n = 5;
std::generate_n(vec.begin(), n, [&x](){return x*=3;}); // 3, 9, ...
```