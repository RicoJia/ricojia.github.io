---
layout: post
title: C++ - Constness
date: '2023-02-01 13:19'
subtitle: constexpr, if constexpr
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

## `constexpr`

`constexpr` in C++ allows expressions to be evaluated at **compile time** rather than runtime. This feature helps optimize programs by computing constant values during compilation and enables creating objects with **immutable values** that the compiler can use in constant expressions. Using constexpr with functions and variables encourages safer and more efficient code by catching errors early and reducing runtime overhead.

Compared to `const`:

- `const` marks data as read-only after initialziation, but its value is not **necessarily** known during compile time.
- `constexpr` provides a stronger compile-time guarantee.
  - `constexpr` variables are always `const`
  - `constexpr` functions and pointers doesn't make everything `const`
- Use `constexpr` in compile-time evaluations:
  - `static_asserts`
  - template parameters

### `constexpr` Functions

A `constexpr` function is NOT a `const` function, **nor does it imply that its parameters or local const are const**. It merely guarantees that when provided with `constexpr` arguments, the function can be evaluated at compile time.

One example is: without `constexpr` function, `static_assert` would throw an error:

```cpp
#include <iostream>

// A constexpr function to compute factorial of n
int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}

int main() {
    // Compile-time evaluation using static_assert, but this would throw an error
    static_assert(factorial(5) == 120, "factorial(5) should be 120");
    return 0;
}
```

With `constexpr`, `static_assert` is happy. The function can be evaluated during runtime too:

```cpp
#include <iostream>

// A constexpr function to compute factorial of n
constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}

int main() {
    // Compile-time evaluation using static_assert
    static_assert(factorial(5) == 120, "factorial(5) should be 120");
    // Runtime usage: even though the function is constexpr,
    // it can be used with values not known until runtime.
    int runtime_value = 6;
    int runtime_result = factorial(runtime_value);
    std::cout << "Factorial of " << runtime_value << " is " << runtime_result << "\n";
    return 0;
}

```

### `if constexpr` [C++ 17] To branch to different code path in compile time meta programming

If you have a code branch to be discarded in compile time, and the condition is available during compile time, if constexpr could help reduce the unecessary code branch.

```cpp
inline size_t hash_function(const NNPoint &pt) {
    if constexpr(dim ==2) {
        return size_t(((pt[0] * 73856093) ^ (pt[1] * 471943)) % 10000000)
    } else if constexpr(dim == 3){
        return size_t(((pt[0] * 73856093)^ (pt[1] * 471943) ^ (pt[2] * 83492791)) % 10000000); 
    }
}
```

### `if Constexpr` & `static_assert`

**On some C++ 20 comiliers (for which I don't know further specifications yet)**, in `if constexpr`, if `static_assert()` is not dependent on any template parameter, it would fire anyways. So one needs to make it explicitly dependent on the desired template parameter so it's reached in the correct code path.

```cpp
  if constexpr (dim == 3)
      query_pt = {pt.x, pt.y, pt.z};
  else if constexpr (dim == 2)
      query_pt = {pt.x, pt.y};
  else
      // BAD: 
      // static_assert(false, "dimension can only be 2 or 3");
      // GOOD: 
      static_assert(dim != 2 && dim != 3, "dimension can only be 2 or 3");
```

However, [on this C++ compiler](https://www.onlinegdb.com/online_c++_compiler), below works fine:

```cpp
#include <iostream>

template<bool true_val>
void g() {
    if constexpr (true_val) {
        // []<bool flag = false>() { static_assert(flag, "static assert"); }();
        // This works properly
        static_assert(true_val, "static assert true val");
        static_assert(false, "static assert false in constexpr true path");
    } else {
    }
}

int main()
{
    g<false>();
    return 0;
}
```

## Constexpr Improvements [C++20]

`constexpr` rules in cpp20 are relaxed. Operations like allocation, resizing, element access on vectors are allowed, as long as those operations themselves meet the requirements for constexpr evaluation

While `vectors` still can't be compiled on my C++20 compiler, one can use `std::array`

```cpp
#include <iostream>
#include <numeric>
#include <array>


int main() {

    std::cout << std::endl;

    constexpr std::array myArray{1, 2, 3, 4, 5};                                     // (1)
    constexpr auto sum = std::accumulate(myArray.begin(), myArray.end(), 0);         // (2)
}
```
