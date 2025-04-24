---
layout: post
title: C++ - [Compilation 4] Name Mangling
date: '2023-05-07 13:19'
subtitle: `nm`
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Motivating Example

```cpp
void foo(){
    const size_t N = 100'000'000;
    std::vector<int> v(N);
    // fill v with 1,2,3,...
    std::iota(v.begin(), v.end(), 1);

    // sum in a tight loop
    long long sum = 0;
    for (size_t i = 0; i < N; ++i) {
        sum += v[i];
    }

    std::cout << "sum = " << sum << "\n";
}

int add(int a, int b) {
    return a + b;
}

int main() {
    foo();
    return 0;
}
```

1. Compile to an object file

```
g++ -c example.cpp -o example.o
```

2. Inspect raw (mangled) symbols `nm example.o`, you might see:

```
0000000000000000 T _Z3foov
000000000000015c T _Z3addii
```

- `_Z3foov` → mangled for foo()
- `_Z3addii` → mangled for add(int,int)



If we compile this file to an object file `example.o`:

```
g++ -C example.cpp
```

2. We can see symbols with `nm`: `nm example.o`:

```
000000000000015c T _Z3barv
0000000000000000 T _Z3foov
```
    
3. Demangle a symbol `echo '_Z3addii' | c++filt`, output: `add(int, int)`

4. Demangle in-place with nm `nm -C example.o`. Example output:

```
0000000000000000 T foo()                 ← foo() at offset 0x0 in .text
000000000000015c T add(int, int)         ← add() at offset 0x15c in .text
                 U __stack_chk_fail      ← undefined symbol
0000000000000000 W __gnu_cxx::new_allocator<int>::allocate(...)
                                        ← weak symbol (address 0 until link)
0000000000000000 r __pstl::execution::v1::seq
                                        ← read-only data at offset 0 in .rodata

```
- `nm` is a tool in GNU `Binutils` that lists symbol table of an `.o`, `.so`, and static library `.a`, or executable
- Symbol types are:
    - `T` = global (external) function in .text
    - `r` = read-only data
    - `W` = weak symbol
- Symbol Interpretation
    - `foo` is the first symbol in this section, so it lives at memory `0000000000000000`
    -  `__gnu_cxx::new_allocator<int>...` is a weak symbol, meaning it has a default value of 0 until the final link decides their address.
    - Once you've linked into an executable or `.so`m these offsets become real address.
    