---
layout: post
title: C++ - Datatypes
date: '2023-01-05 13:19'
subtitle: std::optional, structural binding, iostream, namespace, union, std::variant,  std::type_index, std::any
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

## Namespace

- `::testing::InitGoogleTest(&argc, argv);`: defensive way to avoid accidental namespacing issue. This is to call the gtest module
- Variable name: `_` in C++ is just another variable, not a wildcard.


## Structural Binding
TODO

## `iostream`

c means "character-based stream" in `cin`, `cout`.

- `cout`: This is the standard output stream and **is buffered by default**. This means that data sent to cout is stored in a buffer and is written out (flushed) either **when the buffer is full**, when a newline is encountered (depending on the implementation and whether the stream is tied to an input stream), or when it is explicitly flushed (using `std::flush` or `std::endl`).

- `cerr`: This stream is meant for error messages and **is unbuffered by default**, **so it flushes its output immediately**. This is useful when you want to ensure error messages are output right away.

- `clog`: Similar to cerr in that it’s used for logging messages, but unlike cerr, clog is buffered. This means it collects log messages in a buffer and flushes them less frequently, which can improve performance.

Because clog is buffered, **it can be more efficient for logging non-critical messages** since it reduces the number of flush operations. Downside: if your program crashes before the buffer is flushed, some log messages might be lost. 

There is clog as well. `clog` is faster because it stores output to file / screen in a buffer, once the buffer is full, the buffer flushes. In contrast, `cout` flushes immediately. `clog` is most useful in writing non-essential logs that could be lost upon system crashes.


## `union` and `std::variant`

`union` in C allows you to store different types of variables in the same variable. Those variables share the same memory storage, and only the latest data will get to determine the type of the union. However, there's no way to determine the type of union in runtime, which could be a significant type-safety risk. Meanwhile, **in C++, non-trivial types like `std::string`, references, etc.** cannot be stored in union. To address these limitations, `C++17` introduced `std::variant`(C++ 17)

```cpp
union Foo{
    int x;
    unsigned char c[sizeof(int)];
};
int main()
{
    Foo f;
    f.x = 1234;
    std::cout<<f.x<<std::endl;
    std::cout<<f.c[0]<<std::endl;   //// Undefined: reading an inactive member
    
    return 0;
}
```

`std::variant` is a type-safe tagged union. 

- To initialize, state the variant can only be monostate: `std::variant<std::monostate, bool, int> var`
- `var.index()` gives the current active value
- `std::get<bool>(var)` will throw a `std::bad_variant_access` error if wrong
- `std::get_if<T>(&var)` will return a `nullptr` instead

```cpp
#include <iostream>
#include <variant>

using std::cout; 
using std::endl; 

int main()
{
    std::variant<std::monostate, bool, int> var;
    cout<<var.index()<<endl;    // 0
    var = true;
    cout<<var.index()<<endl;    // 1
    cout<<std::get<bool>(var)<<endl;    // print 1
    var = 42;
    if (auto ip = std::get_if<int>(&var)){
        cout<<*ip<<endl;
    }
    return 0;
}
```

## `std::type_index`

The expression `typeid(T)` returns a `const std::type_info&`, which cannot be copied or assigned.
To store type information in containers or use it as a key (e.g. in `std::unordered_map`), use `std::type_index` from the `<typeindex>` header.

```cpp
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

std::unordered_map<std::type_index, std::string> type_names;
type_names[typeid(int)] = "int";
```

🔹 `std::type_index` is copyable and assignable, making it suitable for associative containers.
🔹 It is only for comparing or storing type_info references — it cannot be used for casting or type deduction.

## `std::any`

Introduced in C++17, `std::any` is a type-erased container that can hold a value of any type, as long as the type is copy-constructible.

```cpp
#include <any>
#include <iostream>
#include <string>

std::any a = 1;
std::cout << std::any_cast<int>(a) << "\n"; // prints 1

a = std::string("Hello");
std::cout << std::any_cast<std::string>(a) << "\n"; // prints "Hello"
```

To retrieve the value, use `std::any_cast<T>(obj)`. If the `type T` does not match the stored type, it throws `std::bad_any_cast`.

```cpp
try {
    std::any a = 1;
    std::any_cast<double>(a); // throws
} catch (const std::bad_any_cast& e) {
    std::cerr << "Bad cast: " << e.what() << std::endl;
}
```
🔹 std::any behaves like a regular value: you can assign, move, and copy it.
🔹 Use it when you need to store values of different types without templates, but still want safe type checking at runtime.