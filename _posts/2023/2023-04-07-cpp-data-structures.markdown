---
layout: post
title: C++ - Datastructures
date: '2024-04-04 13:19'
subtitle: Priority Queue
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - ROS
---

## `std::priority_queue` (Priority Queue, or Max-Heap)

```cpp
#include <iostream>
#include <queue>
#include <vector>
#include <string>

// Custom object
struct Person {
    std::string name;
    int age;
};

// Custom comparator: returns true if p1 should come after p2 (i.e., if p1.age is less than p2.age).
// This ensures that the person with the highest age appears at the top of the max heap.
struct ComparePerson {
    bool operator()(const Person &p1, const Person &p2) {
        return p1.age < p2.age;
    }
};

int main() {
    // Create a max heap (priority queue) of Person objects using our custom comparator
    std::priority_queue<Person, std::vector<Person>, ComparePerson> pq;
    
    // Insert some Person objects into the priority queue
    pq.push(Person{"Alice", 30});
    pq.push(Person{"Bob", 25});
    pq.push(Person{"Charlie", 35});
    pq.push(Person{"Diana", 28});
    
    // The top of the priority queue will be the Person with the highest age.
    std::cout << "Persons in order of descending age:" << std::endl;
    while (!pq.empty()) {
        Person top = pq.top();
        std::cout << top.name << " (" << top.age << ")" << std::endl;
        pq.pop();
    }
    
    return 0;
}
```

- `top()` is O(1), `pop()` is O(log(N))
- A heap requires `operator<`. when `p1 < p2`, the heap is a **max heap**, that is, the top is the maximum value. 


