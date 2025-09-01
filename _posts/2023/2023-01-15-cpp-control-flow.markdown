---
layout: post
title: C++ - Control Flow
date: '2023-01-15 13:19'
subtitle: switch-case, cpp20 range, DSL
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## switch-case

Like `goto`, `switch-case` is a label. **Without `{}`, any variable declared within a switch-case statement has the scope of the entire statement.** This will only work with variables that can be **default-initialized**, because they can be declared only with an intermediate value. This creates a problem: what if a variable that cannot be default initialized is created?

```cpp
// A can be default-initialized
class A {
public:
  A ()=default;
};
class C{
public:
    C(){}   // This is NOT default initialization
};
int main()
{
    int i = 1;
    switch (i)
    {
    case 0:
        A j;    // Compiler implicitly calls default constructor, fine
        int b;  // POD can be declared only, fine.
        int d = 1;  // ERROR: jumping over initialization
        C c;    // ERROR:  jumping over initialization of 'C'
        break;
    case 1:
        b = 2;  // This is assignment, not initialization, fine
        break;
    }
}
```

This is a typical "Jump Over Initialization" error, where initialization may not happen before a variable gets used. (`int d = 1;` is another example, because its initialization may not be executed.)

In this case, the best practice **is to limit the scope of the variable:**

```cpp
case 0:{
    int d = 1;  // fine
    C c;    // fine
    break;
}
```

[Reference](https://stackoverflow.com/a/92730)

### Other Notes On `switch-case`

- `switch-case` only works with integer types.

## Range [C++ 20]

Let's say we are looking to build a data pipeline in a functional-programming style. Along this pipeline, we have multiple conditions for filtering, and multiple transforms to apply. In C++ 17, we might need to create multiple intermediate containers, and while we create them, we might need to copy elements multiple times. Some of those elements however, may not be able to get to the final stage. Here's an example:

```cpp
#include <vector>
#include <algorithm>
#include <iostream>

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5, 6};

    // Filter even numbers => intermediate1
    std::vector<int> intermediate1;
    for (int i : v) {
        if (i % 2 == 0) {
            intermediate1.push_back(i);
        }
    }

    // Multiply by 2 => intermediate2
    std::vector<int> intermediate2;
    for (int i : intermediate1) {
        intermediate2.push_back(i * 2);
    }

    // Filter out any results over 8 => intermediate3
    std::vector<int> intermediate3;
    for (int i : intermediate2) {
        if (i <= 8) {
            intermediate3.push_back(i);
        }
    }

    // Add 1 => final_result
    std::vector<int> final_result;
    for (int i : intermediate3) {
        final_result.push_back(i + 1);
    }

    // Now print final_result
    for (int val : final_result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

However, in C++20, we can chain those filters and transforms together to allow every single element to pass through or get eliminated along the way. At the end stage, we only create one container, and create / copy only the final elements over. This could greatly reduce the amount of memory for container creation and copy times.

```cpp
#include <vector>
#include <iostream>
#include <ranges>     // C++20

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5, 6};

    // Build a lazy pipeline:
    auto pipeline = v
        | std::views::filter([](int i) { return i % 2 == 0; })   // keep evens
        | std::views::transform([](int i) { return i * 2; })     // multiply by 2
        | std::views::filter([](int i) { return i <= 8; })       // keep <= 8
        | std::views::transform([](int i) { return i + 1; });    // add 1

    // Actually iterate (execute) the pipeline:
    for (int val : pipeline) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

- One thing to note is, the pipeline doesn't get run until we actually call the for loop: `for (int val : pipeline)`
- This is similar to Python's `yield`, where data are generated based on the need.

### DSL

A domain-specific language (DSL) is a programming or specification language that's tailored to a particular application domain. Unlike general-purpose languages (like Python, C++, or Java), which are designed to solve a wide range of problems, DSLs are focused on providing specialized notations and abstractions that make it easier to express solutions within a specific area.

- SQL: Used for managing and querying relational databases.
- HTML/CSS: Used for structuring and styling web pages.
- Regular Expressions (regex): Used for pattern matching within text.
- DSL: (Domain Specific Language) is not a C++20 feature. But ranges is very DSL like:

```cpp
auto result = nums
    | std::views::filter([](int i) { return i % 2 == 0; })
    | std::views::transform([](int i) { return i * i; });
```

## `if with initializer` (C++ 17)

{% raw %}

```cpp
#include <iostream>
#include <unordered_map>
int main() {
    std::unordered_map <int, std::string> added_map{
        {1, "one"}, {2, "two"}
    };
    if (auto iter = added_map.find(2); iter != added_map.end()) {
        std::cout << "Found: " << iter->second << std::endl;
    } else {
        std::cout << "Not found" << std::endl;
    }
}
```

{% endraw %}
