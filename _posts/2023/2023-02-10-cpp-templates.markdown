---
layout: post
title: C++ - Templates
date: '2023-02-10 13:19'
subtitle: Class Templates, Dependent Name Lookup, Non-Type Template Parameters, requires, concept, 
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

## Class Template

- Be careful with nested class inside a class template - we need to specify the template parameters here as well.

```cpp
template <typename T>
class A{
public:
    struct B{};
    B b_;
};
int main()
{
    A<int>::B b;
    return 0;
}
```

### Dependent Name Lookup

**`this->` is necessary to call a parent class function in a derived class of a class template.**

```cpp
template <typename T>
class A{
public:
    void foo(){}
};

template <typename T>
class B: A<T>{
public:
    void call_foo(){this->foo();}
};
```

This is because:

- In a regular derived class, we do NOT need `this`. 
- When a derived class derives from a base class, its members' names are called "dependent names". 
- Dependent name lookup is in 2 phases:
    - look up names that do not depend on template params
    - when a template is instantiated, resolve the remaining names
- A parent class function in this case requires template params. We need to specify `this->` to clearly show the compiler.

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

### Specialized Template Function To Provide One Implementation

```
template <>
inline size_t hash_vec<2>::operator()(const Eigen::Matrix<int, 2, 1>& v) const {
    return size_t(((v[0] * 73856093) ^ (v[1] * 471943)) % 10000000);
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

## Concepts For T [C++ 20]

When enforcing type constraints, in C++ 17, one needs to use `std::enable_if`. That results in a longer template signature. In C++ 20, one just needs to use `Concepts`

```cpp
#include <iostream>
#include <type_traits>
#include <concepts>

template <std::integral T>
void print_20(T x) {
    std::cout << x << " is an integer\n";
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, void>::type
print_17(T x) {
    std::cout << x << " is an integer\n";
}

int main(){
    // print_17("123");   // See compiler error: no matching function for call to ‘print(const char [4])’
    print_17(123);     // can compile fine
    print_20(456);
}
```

- ✅ More readable.
- ✅ More efficient (compiler checks constraints upfront).
  - In C++17, `std::enable_if` **is checked during template instantiation**, where errors are usually versbose.
  - In C++20, concepts are checked before template instantiation. Errors are clearer.

## The `requires` Clause [C++20]

Puts constraints on template parameters. It takes a "boolean concept"

- **compile-time predicate** `std::integral<T>` is equivalent to `concept integral = std::is_integral_v<T>;`

```cpp
template<typename T>
T add(T a, T b) requires std::integral<T> {
    return a + b;
}

// Can be before the function signature, or after
template<typename T>
requires std::integral<T>
T add(T a, T b) {
    return a + b;
}
```

- Or, `(dim == 2)`:

```cpp
template <int dim>
requires (dim == 2 || dim == 3)
void foo(int i){}
```

- Commonly, it works with a custom concept. Otherwise, we will have to use `SFINAE`, with `enable_if`, type_traits, which come with lower readability:

```cpp
 template <class T>
 concept Check = requires {
     T().clear();
 };
 ​
 template <Check T>
 struct G {};
 ​
 G<std::vector<char>> x;      // 成功
 G<std::string> y;            // 成功
 // 由于std::array没有clear操作,所以编译失败
 G<std::array<char, 10>> z;   // 失败
```

- `type = std::conditional_t<predicate, Type1, type2>`

```cpp
template<bool Flag>
std::conditional_t<Flag, int, double> getValue() {
    if constexpr (Flag)
        return 42;  // int
    else
        return 3.14;  // double
}

template <int dim>
class NearestNeighborGrid {
public:
    using Point = std::conditional_t<dim == 2, Eigen::Vector2f, Eigen::Vector3f>;
};
```

## Template Alias

One not super commonly used feature in C++ is template alias. 

```cpp
template <typename T>
struct A;

template <typename T>
using B = A<T>;
```

One note is that C++ standard has NOT specified if an template alias is the same as the template itself. On some compilers (e.g., x86-64 clang 10.0.1), the below code snippet can compile fine. But on some others, A and B will be treated as the same template, so the template specialization would yield errors (x86-64 gcc 14.1)

```cpp
template <typename T>
struct A;

template <typename T>
using B = A<T>;

template <template<typename> class Cont>
struct C;

template<>
struct C<A>{};


template<>
struct C<B>{};

int main(){}
```
