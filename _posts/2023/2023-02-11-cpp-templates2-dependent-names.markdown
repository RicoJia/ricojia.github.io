---
layout: post
title: C++ - [Templates 2] Dependent Names
date: '2023-02-10 13:19'
subtitle: Disambiguate Templates
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

## One Example That Explains Dependent Names

In C++, a name is dependent if its meaning or type, value, or definition lookup depends on, a template param `T`. Some examples include:

- `std::vector<T>`: name `std::vector` is not dependent. `std::vector<T>` is dependent
- `std::vector<T>::iterator `(type of `::iterator` depends on T)
- `t.do_something()` (member lookup depends on T)
- `sizeof(T)` (value depends on T)

Also, In C++, template code is compiled in two stages:
1. parsing and template definition: the compiler checks syntax and all non-dependent names directly
2. instantiation: after a set of template args have been provided, 
    - All dependent names are looked up
    - Overloaded resolution is performed, template code are generated

Some of the dependent names require keywords `template` or `typename` so the parser knows how to read the token sequence in the parsing phase. This id due to **grammatical ambiguities**, like:

- `t.do_something<int> ❌`, `<` could also mean "smaller than".

In those cases, we need to specify `template` to disambiguate

```cpp
template <typename T>
struct Foo{
    void do_something(){}
    struct Iterator{};
};

template <typename T>
void call_foo_func(T t){
    t.do_something();  // Works for a regular member function, like Foo<T>::do_something after deduction. Its lookup is in 
}

struct Bar{
    template <typename B>
    void do_something(){}   // this is a member template: the class itself is not a template, but the method is.
};

template <typename T>
void call_bar_func(T t){
    // Works for member templates, to disambiguate (for the sake of '<')
    // Member templates are depedent on T
    t.template do_something<int>();  // Not working on Foo<T>::do_something() because it is not a member template
}

int main()
{
    printf("Hello World");
    Foo<int> f;
    call_foo_func(f);
    Bar b;
    // call_foo_func(b);    // ❌ 
    call_bar_func(b);   
    // call_bar_func(f); ❌ 
    return 0;
}
```

So in the example above, 
- Inside `call_foo_func` the expression t.do_something() is dependent; once `call_foo_func` is instantiated with `Foo<int>` the usual overload resolution chooses the regular member function. There's no ambiguity in parsing.
- `template <typename T> struct Foo` is a class template, `Bar::do_something<T>` is a member template of Bar. 
    - There's an ambiguity in interpreting `<`
    - Therefore, we need to disambiguate by specifying `template` in 't.template do_something<int>();'. 
        - Otherwise, in parsing, `do_something<int>()` could be interpreted as "member attribute do_something is smaller than int" , which raises a parsing error. 

## Disambiguate Type With Typename

When there are grammatical ambiguity of **dependent types on T**, we need to specify `typename` to the compiler :

```cpp
template <typename T>
void call_func(T t){
    typename std::vector<T>::iterator it;
}
```

Do you see why `std::vector<T>::iterator` needs to be disambiguated? Because `iterator` could be also interpreted a static variable:

```cpp
template <typename T>
struct Weird {
    static int iterator;           // could exist!
    using iterator_type = int;
};
```

