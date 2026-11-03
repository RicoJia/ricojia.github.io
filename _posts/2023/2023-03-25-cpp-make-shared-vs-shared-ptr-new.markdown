---
layout: post
title: C++ - [Pointers - 3] - make_shared vs shared_ptr(new)
date: '2023-03-25 13:19'
subtitle: A hidden memory tradeoff of make_shared when weak_ptr is involved
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

In C++, there are two common ways to create a `std::shared_ptr`:

```cpp
auto p1 = std::shared_ptr<Widget>(new Widget());
auto p2 = std::make_shared<Widget>();
```

Both provide shared ownership of a `Widget` object. However, they differ in how memory is allocated, and this difference becomes important when `std::weak_ptr` is involved.

## 1. How Does `shared_ptr` Work?

A `shared_ptr` relies on two components:

- **The object** (`Widget`): The actual data we want to manage.
- **The control block**: Stores ownership information, including the number of shared and weak references.

The control block typically maintains two reference counts:

- **Strong count:** Tracks shared ownership through `shared_ptr`.
- **Weak count:** Tracks weak references through `weak_ptr` (with some implementation-specific bookkeeping).

When the last `shared_ptr` disappears, the managed object is destroyed. However, the control block must remain alive as long as any `weak_ptr` still references it.

## 2. The Difference in Memory Allocation

### Method A: `shared_ptr(new Widget())`

```cpp
auto p = std::shared_ptr<Widget>(new Widget());
```

This typically involves two separate memory allocations:

```
Allocation 1:
+------------------+
|   Widget object  |
+------------------+

Allocation 2:
+------------------+
|   Control block  |
|  - Strong count  |
|  - Weak count    |
+------------------+
```

Because these allocations are separate, **the object's memory can be released independently** of the control block.

### Method B: `make_shared<Widget>()`

```cpp
auto p = std::make_shared<Widget>();
```

Most standard library implementations combine the object and control block **into a single allocation**:

```
Single allocation:
+------------------+
|   Widget object  |
+------------------+
|   Control block  |
|  - Strong count  |
|  - Weak count    |
+------------------+
```

This has two main advantages:

1. **Better performance:** Only one memory allocation is typically required.
2. **Lower memory overhead:** Fewer allocations can mean less allocator overhead and better memory locality.

This is why `std::make_shared` is generally recommended.

But there is a subtle tradeoff.

## 3. What Happens When `weak_ptr` Is Involved?

Consider the following example:

```cpp
#include <memory>
#include <iostream>

struct Widget {
    char data[10 * 1024 * 1024]; // 10 MB

    ~Widget() {
        std::cout << "Widget destroyed\n";
    }
};

int main() {
    std::weak_ptr<Widget> weak;

    {
        auto p = std::make_shared<Widget>();
        weak = p;
    }

    // The last shared_ptr has disappeared.
    // Widget's destructor has been called.
    // But the combined allocation is still alive.

    std::cout << weak.expired() << "\n"; // true

    weak.reset();

    // Now the combined allocation can be released.
}
```

Let's examine what happens.

**Step 1: Create a `shared_ptr`.**

The object and control block are allocated together.

```
Strong count = 1
Weak count   = 0
```

**Step 2: Create a `weak_ptr`.**

```cpp
weak = p;
```

The weak count increases, but the strong count does not.

```
Strong count = 1
Weak count   = 1
```

**Step 3: The `shared_ptr` goes out of scope.**

The strong count becomes zero.

```
Strong count = 0
Weak count   = 1
```

At this point:

- The `Widget` destructor executes.
- The object is no longer alive.

- The control block remains alive because the `weak_ptr` still exists.
- **The 10 MB combined allocation cannot yet be released.**

This last point is the hidden cost of `make_shared`.

Even though the object's lifetime has ended, its storage remains allocated.

**Step 4: The `weak_ptr` is destroyed.**

```cpp
weak.reset();
```

There are now no remaining references.

The control block is destroyed, and the combined allocation can finally be released.

## 4. What If We Use `shared_ptr(new Widget())` Instead?

Consider the same example:

```cpp
std::weak_ptr<Widget> weak;

{
    auto p = std::shared_ptr<Widget>(new Widget());
    weak = p;
}

// Widget is destroyed.
// Widget's 10 MB allocation has been released.
// The control block still exists.

weak.reset();

// The control block is now released.
```

Because the object and control block were allocated separately, the object's 10 MB allocation can be released immediately after the last `shared_ptr` disappears.

The `weak_ptr` **keeps only the control block alive**.

## 5. Comparison

| Behavior                                                         | `make_shared` | `shared_ptr(new)`        |
| ---------------------------------------------------------------- | ------------- | ------------------------ |
| Typical memory allocations                                       | 1             | 2                        |
| Allocation performance                                           | Better        | Slightly worse           |
| Object destroyed when strong count reaches 0                     | Yes           | Yes                      |
| Control block retained by `weak_ptr`                             | Yes           | Yes                      |
| Object's allocated storage released while weak references remain | No            | Yes                      |
| Typical recommendation                                           | Preferred     | Useful in specific cases |

## 6. An Important Detail: Object Memory vs. Dynamically Allocated Memory

Suppose the `Widget` instead uses a `std::vector`:

```cpp
struct Widget {
    std::vector<char> data;

    Widget() : data(10 * 1024 * 1024) {}
};
```

Here, the 10 MB buffer is allocated separately by the vector.

When the `Widget` destructor executes, the vector's destructor also executes, releasing its buffer. Therefore, even if `make_shared` keeps the object's combined allocation alive, the vector's 10 MB buffer can still be released. This distinction matters:

- **Memory stored directly inside the object** may remain allocated.
- **Memory dynamically owned by the object** is normally released by its destructor.

## 7. Which Method Should We Use?

In most applications, `std::make_shared` is the better choice because it reduces allocation overhead and is convenient to use.

However, `std::shared_ptr(new Widget())` may be worth considering when:

- The object itself occupies a large amount of memory.
- **Many long-lived `weak_ptr` instances** refer to objects after their shared ownership has ended.

- Releasing the object's storage promptly is important.

Also **remember that a `weak_ptr` does not keep an object alive**. It only keeps the control block available so that the program can safely determine whether the object still exists.

The most important lesson is that **object destruction and memory deallocation are not necessarily the same event**.
