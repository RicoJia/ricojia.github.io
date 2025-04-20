---
layout: post
title: C++ - Move Semantics, Perfect Fowarding
date: '2024-04-10 13:19'
subtitle: Universal Reference, `std::move` with Emplace
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - ROS
---

## Universal Reference And Perfect Forwarding

Perfect forwarding in C++ preserves both the **type** and the **value category** of **an input argument**.

Refresher - Value categories:
- lvalue
- rvalue
    - xvalue (eXpiring)
    - prvalue (pure)

### Universal Reference

`std::forward` calls a callable be it a lvalue, or an rvalue. When a type is deduced in a template, if it's in the form `&&`, it's a universal reference, `Func&&` in this case is the **universal reference** to `func`.  So if it's:

- an lvalue: `profile_and_call(another_function)`, `another_function` is `another_function&`, and `Func&&` is deduced to `another_function& &&` and becomes `another_function&`
- an rvalue: `profile_and_call(std::move(another_function))`, `another_function` is `another_function&&`, and `Func&&` is deduced to `another_function&& &&` and becomes `another_function&&`

Specifically, this preserves the value category of the function, specifically when the function inside has move semantics. If it's a simple `const Func& func` argument, we will not be able to move. Also, if the function is a captured lambda, it's a temporary object that will be best used if it's "moved"

On the other hand, note that **T&& is a universal reference only when T is a deduced template parameter.** (so it doesn't happen outside of the template scenario).

```cpp
void func(int&& arg);   // rvalue reference only
```

The basic format is to declare `Func` in template, then forward it: `std::forward<TYPE>`

```cpp
template<typename Func>
void profile_and_call(Func&& func){
    std::forward<Func>(func)(); // perfectly forward and invoke
}

// Using copy ctor:
template<typename Func>
void profile_and_call(const Func& func){
    func();
}
```

- One rule in C++ is **"named parameters are lvalues"**: inside `void profile_and_call(Func&& func)`,
    - `func` itself is an **lvalue**. we need `std::move()`. or `std::forward` to cast it to the correct value category.

## Need Member Template For Class Template

Inside a class template you add a member template so you can perfectly‑forward whatever key/value pair the caller gives you.

```cpp
template <typename Key, typename Value>
class HashMap{
    ...
    template <typename K, typename V>
    void add(K&& key, V&& value) {
        emplace(std::forward<K>(key), std::forward<V>(value));
    }
};
```

- The compiler will have to deduce K, and V and ensures matching with Key and Value

```cpp
itr_lookup_.find(key)
```

## `std::move` and Double Moved-From Issue

A "moved-from" state is a valid yet unspecified state. You can assign another value to it, destroy it, etc. 

```cpp
#include <iostream>
#include <string>
#include <utility>

struct Tracer {
    std::string name;

    Tracer(std::string n): name(std::move(n)){
        std::cout << "[Ctor]   name = \"" << name << "\"\n";
    }

    Tracer(const Tracer& other) : name(other.name) {
        std::cout << "[Copy]   name = \"" << name << "\"\n";
    }

    Tracer(Tracer&& other) noexcept : name(std::move(other.name)){
        std::cout << "[Move]   new.name = \"" << name
                  << "\",  old.name = \"" << other.name << "\"\n";
    }
};

template <typename T>
void double_move(T&& x) {
    std::cout << "-> entering double_move, x.name = \"" << x.name << "\"\n";

    // First move-from
    Tracer a(std::forward<T>(x));   // should see a's move ctor called, x name is valid
    std::cout << "   after first move, x.name = \"" << x.name << ", a name"<<a.name<<"\"\n";

    // Second move-from
    Tracer b(std::forward<T>(x));   // should see b's move ctor called, but x's name already is ""
    std::cout << "   after second move, x.name = \"" << x.name << ", b name"<<b.name<< "\"\n";
}

int main() {
    Tracer t("original");
    double_move(std::move(t));
    std::cout << "-- back in main, t.name = \"" << t.name << "\" --\n";
    return 0;
}
```

- The output shows only move-ctors are called, no copy ctor is called. But the first `std::forward` call has nullified x into a moved-from state, which would be an un-defined state
- **`std::move()` is designed to move an lvalue reference into an rvalue reference**. `double_move(std::move(t));` makes `t` bind to  `T&&`, enables perfect forwarding, so move ctor can be called later with `std::forward<TYPE>()` or `std::move<TYPE>()`. If `double_move(t);` is used, everything will be copied. 


### `prvalue`

TODO

```cpp
auto map_itr = itr_lookup_.find(key);   // this is a pr-value
auto& map_itr = itr_lookup_.find(key);
```

## Emplace with `std::move`

```cpp
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <unordered_map>

//----------------------------------------------------------------------
// A small tracer for keys
struct KeyTracer {
    std::string name;

    KeyTracer(const std::string& n) : name(n) { std::cout << "[Key Ctor]    \"" << name << "\"\n"; }
    KeyTracer(const KeyTracer& o) : name(o.name) { std::cout << "[Key Copy]    \"" << name << "\"\n"; }
    KeyTracer(KeyTracer&& o) noexcept : name(std::move(o.name)) { std::cout << "[Key Move]    new=\"" << name << "\"  old=\"" << o.name << "\"\n"; }
    // equality for unordered_map
    friend bool operator==(KeyTracer const& a, KeyTracer const& b) {
        return a.name == b.name;
    }
};

//----------------------------------------------------------------------
// Provide std::hash<KeyTracer>
namespace std {
  template<> struct hash<KeyTracer> {
    size_t operator()(KeyTracer const& k) const noexcept {
      return std::hash<std::string>()(k.name);
    }
  };
}

//----------------------------------------------------------------------
// A small tracer for values
struct ValueTracer {
    int x, y;

    ValueTracer(int _x, int _y) : x(_x), y(_y) {std::cout << "[Value Ctor]  (" << x << ", " << y << ")\n"; }
    ValueTracer(const ValueTracer& o) : x(o.x), y(o.y) { std::cout << "[Value Copy]  (" << x << ", " << y << ")\n"; }
    ValueTracer(ValueTracer&& o) noexcept : x(o.x), y(o.y) { std::cout << "[Value Move]  new=(" << x << ", " << y << ")  old=(" << o.x << ", " << o.y << ")\n";}
};

int main() {
    std::unordered_map<KeyTracer,ValueTracer> m;

    std::cout << "\n--- emplace with piecewise_construct ---\n";
    // construct key and value right on the spot, zero temp, zero move. MOST EFFICIENT
    m.emplace(
        std::piecewise_construct,
        std::forward_as_tuple("apple"),    // builds key in-place
        std::forward_as_tuple(7,  42)                   // builds value in-place
    );  // std::piecewise_construct is needed. 
    
    m.emplace(
        KeyTracer{"apfel"},      // this constructs a KeyTracer
        ValueTracer{7,42}        // this constructs a ValueTracer
    );  // emplace will 1. construct a temp Key and Value object. 2. forwards those temps into std::pair, which moves each once.
    
    // m.emplace({"apfel"}, {7,42});    // Doesn't compile because emplace(Args&&) will foward them. 
    return 0;
}
```

- In unordered map, `m.insert(std::pair)` takes in a pair. 
- As in multiple containers (like std::vector), `m.emplace(Args&&)` takes in arguments to construct elements that can go into `m.insert()` or `m.push_back()`
- `emplace`:
    - `std::piecewise_construct` uses `std::pair`'s piecewise ctor

        ```cpp
        m.emplace(
            std::piecewise_construct,
            std::forward_as_tuple("apple"),    // → KeyTracer(const char*)
            std::forward_as_tuple(7, 42)       // → ValueTracer(int,int)
        );
        ```
        - `std::forward_as_tuple(arg1, arg2...)` is just a convenient way to create a **tuple of universal references to the args**.
        - So `std::forward_as_tuple(7, 42)` goes into a `ValueTracer(int,int)` ctor

    - Without `std::piecewise_construct`, temps are constructed first, then moved into a `std::pair`:

        ```cpp
        m.emplace(
            KeyTracer{"apfel"},      // this constructs a KeyTracer
            ValueTracer{7,42}        // this constructs a ValueTracer
        );  // emplace will 1. construct a temp Key and Value object. 2. forward and calls move ctor on the temps  
        ```

- `m.emplace{{ "apfel" }}` fails because:
    - **A function template like `template <class... Args> pair<iterator,bool> emplace(Args&& args);` needs clear typenames, and won't take in braced-init list {}**
- `try_emplace{Args&&}` is introduced in C++17 that in-place constructs the value **only if the key isn’t already present; otherwise does nothing.**. It's **idempotent**