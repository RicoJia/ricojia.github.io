---
layout: post
title: C++ - Container Operations
date: '2023-01-28 13:19'
subtitle: Vector, Map, Algorithms
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Common Operations

### Insert 

- Insert Or Assign (C++17)

```cpp
#include <iostream>
#include <unordered_map>
int main() {
    std::unordered_map<int, std::string> my_map;

    my_map.insert_or_assign(1, "apple");   // Insert key 1
    my_map.insert_or_assign(2, "banana");  // Insert key 2
    my_map.insert_or_assign(1, "avocado"); // Assign new value to key 1
    std::cout << "After: " << my_map[1] << "\n";   // Outputs: avocado
}
```

In set and unordered_set, if an element already exists, insert does not do anything:

```cpp
#include <iostream>
#include <set>
int main() {
    std::set<int> my_set;

    auto result1 = my_set.insert(42);  // First insert
    auto result2 = my_set.insert(42);  // Second insert
    std::cout << "Set size: " << my_set.size() << "\n";               // 1
}
```

### Emplace

- `unordered_map::emplace() -> pair [iterator, bool]` vs `try_emplace()`: 
    - `try_emplace()` Only constructs the value if the key doesn’t exist.
    - `emplace(key, value)` Always constructs the key and value (even if the key already exists — it won’t be inserted though). 
        - Returns a pair with an iterator and a bool (false if the key already existed).

```cpp
#include <iostream>
#include <unordered_map>
#include <string>

int main()
{
    std::unordered_map<std::string, std::string> my_map;
    std::string expensive = "expensive";
    
    // emplace will always create the pair, even if key exists
    auto it = my_map.emplace("dog", expensive);  
    
    // try_emplace avoids creating the value if "dog" is already in map
    my_map.try_emplace("dog", "dog2");  // Only constructs the value if the key doesn’t exist.
    // always construct the key-value pair, but if the object exists, the pair won't be inserted
    my_map.emplace("dog", "expensivo");  
    std::cout<<my_map["dog"]<<std::endl;    // see expensive
    std::cout<<(it.first->second)<<std::endl;    // see expensive
    return 0;
}
```



### Erase:

{% raw %}
```cpp
// 
std::map<int, std::string> m = {{1, "one"}, {2, "two"}};
m.erase(1);  // Removes key , o(log n)
std::unordered_map<int, std::string> umap = {{1, "one"}, {2, "two"}};
umap.erase(1);  // Removes key , o(1)
std::set<int> s = {1, 2, 3};
s.erase(2);  // Removes element with value 2, o(log n)
std::unordered_set<int> uset = {1, 2, 3};
uset.erase(2);  // Removes element with value 2 o(1)

// Alternative 2: universal erase with iterator:

auto it = my_map.find(key);
if (it != my_map.end()) {
    my_map.erase(it);  // faster than key lookup + erase
}
```
{% endraw %}

## Vector

- Append `vector2` to the end of `vector1`

```cpp
vector1.insert(vector1.end(), vector2.begin(), vector2.end());
```

- `nodes_.resize(cloud->points.size());` the new elements are value-initialized,


## Associative Containers

### Hashing

By default, keys of types `std::pair<int, int>` are not hashable. One needs to define a callable for that.

The first alternative is defining a functor

```cpp
#include <iostream>
#include <unordered_map>
#include <utility> 

struct MyHash{
    size_t operator()(const std::pair<size_t, size_t>& p) const {
        return std::hash<size_t>{}(p.first) ^ (std::hash<size_t>{}(p.second) << 1);
    }
};

int main() {
    std::unordered_map<std::pair<size_t, size_t>, std::string, MyHash> my_map;

    my_map[{1, 2}] = "one-two";
    my_map[{3, 4}] = "three-four";

    std::cout << "Value at (1, 2): " << my_map[{1, 2}] << "\n";
    std::cout << "Value at (3, 4): " << my_map[{3, 4}] << "\n";

    return 0;
}
```

Another alternative is defining a global hash function. It's a bit intrusive

```cpp
#include <unordered_map>
#include <utility>
#include <iostream>
namespace std{
    template<>
    struct hash <std::pair<size_t, size_t>>{
        size_t operator()(const std::pair<size_t, size_t>& p) const {
            return std::hash<size_t>{}(p.first) ^ (std::hash<size_t>{}(p.second) << 1);
        }
    };
}

int main() {
    std::unordered_map<std::pair<size_t, size_t>, std::string> my_map;
    my_map[{7, 8}] = "seven-eight";
    std::cout << my_map[{7, 8}] << std::endl;
}
```


## Sets

## Algorithms

### Nth Element

`nth_element(first, nth, last, comp)` makes sure:

1. At `nth` place of the container, the element is actually the `nth` as if the container were sorted.
2. Elements before the nth element has `comp(i, n) = False`, or without `comp`, they are smaller than the `nth` element

```cpp
// elements before the returned iterator has a value less than or equal to the value
std::nth_element(keypoints.begin(),keypoints.begin() + desired_features_num - 1, keypoints.end(),
    [](const KeyPoint& k1, const KeyPoint& k2){
        //descending 
        return k1.response > k2.response;
    }
);
```

### Partition

`std::partition(first, last, pred)` moves all elements that makes pred(item) **true** to  first iterator, and returns an iterator that points to the first item that makes pred(item) **false**. In combination wtih `std::nth_element`, we can partition and find the iterator that makes sure all elements smaller than or equal to the `nth` elements are before a returned iterator, `new_end`.

```cpp
// we might still have elements equal to the nth element. So, we use partition to find them
// std::partion moves items that satisfies the pred to before the iterator
auto new_end = std::partition(keypoints.begin() + desired_features_num, keypoints.end(), [&keypoints](const KeyPoint& k){k.response == (keypoints.begin() + desired_features_num - 1)->response});
```

## PMR: Polymorphic Memory Resources (C++17)

- Custom Allocators: Instead of having containers decide how memory is allocated, PMR allows you to supply a custom memory resource (allocator) that the container uses.
- Reusable Strategies: You can create memory resources that implement different allocation strategies, such as pooling, monotonic allocation, or synchronized (thread-safe) allocation, and then reuse them across multiple containers.
    - Memory Efficiency: PMR can help you pre-allocate a large chunk of memory (from a buffer) and then use it for many small allocations, which is useful in real-time or embedded systems.
    - Multiple PMR-based containers can share the same memory resource. 
- memory allocation strategies by using user-supplied memory resources.
