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
