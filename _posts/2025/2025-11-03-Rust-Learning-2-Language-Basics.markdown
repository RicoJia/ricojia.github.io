---
layout: post
title: "[Rust Learning]-2-Variables"
date: 2025-11-03 13:19
subtitle: Ownership
header-img: img/post-bg-o.jpg
tags:
  - Python
comments: true
---
## Basic Hello World

In this example, you will see the usage of immutable vs mutables, how to declare variable types, how to pass in args and return from a function.  

```rust
fn main() {
    // type is automatically deduced to i32; 
    // By default, variables are immutables. So you can't do a = 3 after
    let a = 1;
    let b = 2; 
    // {} is format placeholder 
    // println is a Macro, which copies a code block over
    println!("Hello, world! {} + {} = {}", a, b, add(a,b));
    
    // You can declare type.  
    // for f64, you can't do c = 3; 
    let c: f64=3.0;
    
    // FYI, you can declare type in these ways too: 
    let d = 40_i32;
    let e = 50.0f64;
    
    // You can declare a mutable this way; 
    // But its later values must match with the current type
    let mut m=4;
    m=5;
    m=6.0;
}

// Why this function can be defined after the main?
fn add(i: i32, j: i32) -> i32{
    return i + j;
}
```

- **One gotcha is Rust does NOT require top-to-bottom function declaration**. The compiler will first build an index of items in that module. So `main` can call `add` even if `add` appears later in the file
