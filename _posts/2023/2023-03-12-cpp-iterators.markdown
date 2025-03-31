---
layout: post
title: C++ - Iterators
date: '2023-03-06 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Basic Usage

An iterator needs to be dereferenced `*it` or `it->first` so its internal data can be accessed.

If you have an `std::unordered_map<Key, Value>` (or `std::map<Key, Value>`), its iterator points to a key-value pair, represented as `std::pair<const Key, Value>`.

```cpp
auto it = my_map.find("some_key");
std::cout << (*it).first << " : " << (*it).second << std::endl; // see "some key", and value
std::cout << it->first << " : " << it->second << std::endl;
```
- `std::set`, `std::unordered_set`: Iterator points directly to the value (the key itself).

```cpp
std::unordered_set<int> s;
auto it = s.begin();
int value = *it;
int value2 = it.operator*();
int value3 = *it;         // same thing
int value4 = it->second;  // ❌ doesn't work! No .second — it's just a value
```

- `std::vector<T> / std::deque<T> / std::list<T>`:

```cpp
std::vector<int> v;
auto it = v.begin();
*it
```

- std::vector<std::pair<>> or similar

```cpp
std::vector<std::pair<int, std::string>> vp;
auto it = vp.begin();
it->first    // valid!
it->second   // also valid!
```

- `std::tuple` in containers

```cpp
std::vector<std::tuple<int, std::string>> vt;
auto it = vt.begin();
std::get<0>(*it);  // get the first item
std::get<1>(*it);  // get the second item
```