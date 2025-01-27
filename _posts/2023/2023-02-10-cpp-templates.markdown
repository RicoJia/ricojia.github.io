---
layout: post
title: C++ - Templates
date: '2023-02-10 13:19'
subtitle: Non-Type Template Parameters
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

## `typename` Is Required By "Dependent Names"

Dependent names are names that are dependent on the **type** of a template parameter. To instantiate instances of them, or to simply use them as types, one needs to use the keyword **typename**.

Consider the below snippet:

```cpp
// namespace MyClass {  // can't be namespace, because typename T expects a type, such as class/struct
struct MyClass {
    struct constant {
        static constexpr int value = 42;
    };
};

template <typename T>
void exampleFunction() {
    typename T::constant myConstant; // No 'template' keyword needed
    // value is a static member, and it's accessible to the class itself
    std::cout << "Value: " << myConstant.value << std::endl;
}
```

- `::` means type within a type
- `typename T` can't take in a namespace. It has to be a type. It is to **create an object of the type T**
- `typename T::constant myConstant` can access static members, like `myConstant.value`

## Non-Type Template Parameters

Non Type template parameters are compile time constants that customizes a struct. Some notes include:

- `Foo<9>` is a different instantiation of `Foo<10>` 

```cpp
template <int N>
struct Foo{
    static const int i = N;
};

int main()
{
    Foo<9> f;
    std::cout << f.i << std::endl;
    // Print the static class member
    std::cout << Foo<10>::i << std::endl;
}
```

## `template` Keyword Is Required To Disambiguate

Consider the below snippet:

```cpp
#include <iostream>
#include <type_traits> // For std::remove_pointer

template <class T>
void f(T &x) {
    // Resolve the base type of T if T is a pointer
    using BaseType = typename std::remove_pointer<T>::type;
    // So BaseType is FTemplates

    // template function variable() needs to be accessed with an explicit "template"
    // BaseType::template is needed to indicate constant is a nested template inside Basetype
    x->template variable<typename BaseType::template constant<3>>();
}

struct FTemplates {
    // Nested template
    template <int N>
    struct constant {
        static constexpr int value = N;
    };

    // Member template
    template <typename T>
    void variable() {
        std::cout << "variable: " << T::value << std::endl;
    }
};

int main() {
    FTemplates f_tmp;
    FTemplates *f_ptr = &f_tmp;
    // Call f with a pointer to f_templates
    f(f_ptr); // Outputs: variable: 3
    return 0;
}

```