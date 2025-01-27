---
layout: post
title: C++ - Static Functions And Keyword
date: '2023-02-06 13:19'
subtitle: Static Functions And Keyword
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

As a very high level summary, the keyword `static` has two uses: 

- File scope static variables and functions can be only accessed within the current translation unit (i.e., the `cpp` source file)
- Class static variables and functions belong to the class and do not need to be accessed through any class object.

## Static Variable

### Static Variable In Function Scope

Within a function, a static variable is **initialized only once**, and its value is retained across function calls.

```cpp
void num_dec_inside_function(){
    // This is initialized once and persists across function calls
    static int another_num = 1;
    another_num ++;
    std::cout<<"another num: "<<another_num<<std::endl;
}
num_dec_inside_function();  // another num: 2
num_dec_inside_function();  // another num: 3
```

### Static Variable In Global Scope

[Please see my article on C++ linkage.](./2023-01-30-cpp-linkage.markdown)

### Static Member Variable Within A Class

Static member variable is similar to Python's class variable. That is, the variable can be variable without instantiating a object of the class. 

```cpp
class Foo{
    public: 
        // This is declaration, see below for definition
        static int i; 
        void print_i(){
            std::cout<<"i: "<<i<<std::endl;
            ++i;
        }
};

// This is how to initialize static member variable
int Foo::i(123); 

int main(){
    Foo f;
    // see 123
    f.print_i();
    // see 124
    std::cout<<"class variable: "<<Foo::i<<std::endl;

}
```

- Non-static functions can access static members. **One does not need to specify the class name!**
- Static members must be defined outside of the class in the same namescope
- If the static member variable is defined in hpp, then every instance will get the same value. Otherwise, it should be defined in the **source file**.
- Regular member functions can access a static member variable with the scope specifier `::`.

## Static Function

### Static Function In Class Scope

Similar to a static variable, a static member function belongs to the class, instead of to a specific object.

```cpp
class Foo{
    public: 
        // This is declaration, see below for definition
        static int i; 
        // can declare static func inside class
        static void foo_static(){
            cout<<__FUNCTION__<<endl;   // Outputs: foo_static
        }
        
        void foo(){
            foo_static();
            // no need to add class name since both functions are in the same class
            // Foo::foo_static();
        }
};

// See Foo::foo_static();
Foo::foo_static();
```

### Static Function In Global Scope

A static function function defined at the file level has internal linkage, meaning it's limited to the file in which it's defined. So each file will have to define their own version of the function should it needs this function.

```cpp
// foo.hpp
// Comment out the function declaration
// void test_bar(); 

// using static to restrict this function to be accessible only in the current source file 
static void test_bar(){
    std::cout<<"foo funcs test bar"<<std::endl; // foo funcs test bar
}

// using static to restrict this function to be accessible only in the current source file
// and avoid potential naming conflict
static void test_bar(){
    std::cout<<"test bar: main"<<std::endl;
}

int main(){
    test_bar(); // test bar: main
}
```

Note that defining a static function in a header file syntatically is fine, as long as the each translation unit defines its own version of the function. However, without `static`, a function **by default has external linkage which could be shared**. defining a static function in a header file would unnecessarily expose the function signature and is **NOT a common practice**.
