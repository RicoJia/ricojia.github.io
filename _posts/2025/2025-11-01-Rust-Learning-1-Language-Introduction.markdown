---
layout: post
title: "[Rust Learning]-1-Language Introduction"
date: 2025-11-01 13:19
subtitle: Why Rust
header-img: img/post-bg-o.jpg
tags:
  - Python
comments: true
---
# Rust Introduction: A Modern Systems Programming Language

Rust is a systems programming language designed for **performance**, **memory safety**, and **concurrency** — without relying on a garbage collector.

## 🚀 A Brief History

- First stable release: **2015**

- Designed for:

  - **C/C++-level performance**
  - **Memory safety without garbage collection**
  - A **strong, expressive type system**
  - 🔒 Compile-time race prevention
  - **Deterministic resource cleanup**
  - **Race-free concurrency**
  - Modern tooling (`cargo`, `rustfmt`, `clippy`, `rustdoc`)

# Performance Without Garbage Collection

Rust does **not** use a garbage collector. Instead, it ensures memory safety through:

- **Ownership**
- **Borrowing**
- **RAII (Resource Acquisition Is Initialization)**

When a value goes out of scope, it is automatically cleaned up.

```rust
{  
    let s = String::from("hello");  
} // s is dropped here automatically
```

# Ownership & Move Semantics

Rust enforces **single ownership**:

- Each value has exactly **one owner**.
- When the owner goes out of scope, the value is dropped.
- Assignments move ownership by default.

```rust
let s1 = String::from("hello");  
let s2 = s1; // move  
// s1 is now invalid
```

After the move, `s1` can no longer be used. If you really need a copy, you can explicitly clone:

```rust
let s2 = s1.clone(); // deep copy
```

⚠️ Be careful with `.clone()` — it can be expensive since it performs a deep copy.

# Generics & Monomorphization

Rust does not use traditional runtime polymorphism like some OOP languages. Instead, it uses:

- **Traits** (similar to interfaces/typeclasses)
- **Monomorphization**

### What is Monomorphization?

Monomorphization means the compiler generates a specialized version of a generic function for each concrete type used.

Example:

```rust
fn add<T: std::ops::Add<Output = T>>(a: T, b: T) -> T {  
    a + b  
}  
  
fn main() {  
    let x = add(5, 10);       // i32 version generated  
    let y = add(1.5, 2.5);    // f64 version generated  
}
```

The compiler effectively creates separate implementations for `i32` and `f64`. This gives:

- Zero runtime cost
- C++-style template performance
- Strong type guarantees

# Concurrency Without Data Races

Rust guarantees:

> If your program compiles, it does not contain data races.

Key rules:

- Shared data must be synchronized.
- Types must implement:
  - `Send` → safe to transfer between threads
  - `Sync` → safe to reference from multiple threads

These guarantees are enforced at compile time.

# Tooling & Ecosystem

Rust ships with excellent tooling out of the box:

### 📦 Cargo

- Official package manager and build tool
- Dependency resolution
- Lock files (`Cargo.lock`)
- Reproducible builds
- Integrated testing & benchmarking

### 📚 crates.io

- Official package registry
- Large ecosystem
- Many small, composable libraries

One downside:

- Deep dependency trees
- Many small crates
- Can increase compile times

But the ecosystem strongly favors composability and modular design.

---

# Documentation with `rustdoc`

Rust uses doc comments to generate documentation automatically.

```rust
/// Adds two numbers together.  
fn add(a: i32, b: i32) -> i32 {  
    a + b  
}
```

Running:

```bash
cargo doc --open
```

Generates beautiful HTML documentation directly from your source code.

---

## What Rust is NOT designed for

Rust is a multi-paradigm systems language, but it is not _designed around_ classic OOP or pure FP principles the way Java or Haskell are.

Rust has:

- `struct`
- `impl`
- methods
- encapsulation (via `pub`)
- traits (like interfaces)

But Rust **does not** have:

- inheritance
- class hierarchies
- traditional subtype polymorphism by default

Rust prefers:

- **composition over inheritance** (I love this)
- **traits over class hierarchies**
- **static dispatch over runtime polymorphism** (I love this)
