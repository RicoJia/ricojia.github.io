---
layout: post
title: C++ - Weird & Interesting Stuff I found
date: '2023-01-04 13:19'
subtitle: Oh boy, C++ is weird ... but also interesting :)
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Strings

- Raw C strings (char arrays) should always terminate with a **null character**, '\0'. Fun fact about char arrays: `nullptr` is inserted at the back of the array. This is a convention for `argv`

```cpp
char str[] = "a string";
printf("%s", str);  // searches for "\0" to terminate

char *strs[] = {"str1", "str2", nullptr};
for (size_t i = 0; strs[i] != nullptr; ++i){
    // uses nullptr to terminate string array.  
    printf("%s", strs[i]);
}
```

## `if Constexpr` & `static_assert`

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

why?