---
layout: post
title: C++ - Control Flow
date: '2023-01-15 13:19'
subtitle: switch-case
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

