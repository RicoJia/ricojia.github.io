---
layout: post
title: C++ - Common General Sofware Design Pricinples
date: '2023-12-15 13:19'
subtitle: Coupling & Cohesion
comments: true
tags:
    - C++
    - Software Developer's Career Tips
---

## "Plug and Play": Low Coupling, High Cohesion
 
In software engineering, low coupling and high cohesion are two fundamental principles that lead to better code organization, maintainability, and scalability.

- Low Coupling refers to minimizing dependencies between different modules or components. This ensures that changes in one part of the system have minimal impact on others, making the code easier to modify, test, and debug.
- High Cohesion means that each module or class is responsible for a well-defined task and contains related functionalities. This improves code clarity, reusability, and robustness.

By following these principles, developers create modular, flexible, and easier-to-maintain systems, reducing technical debt and making future enhancements smoother.



```cpp
#include <iostream>

// High Cohesion: Class handles only shape-related behavior
class Rectangle {
private:
    double width, height;
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    double area() const { return width * height; }
};

// Low Coupling: Separate class for displaying information
class Display {
public:
    static void printArea(const Rectangle& rect) {
        std::cout << "Area: " << rect.area() << std::endl;
    }
};
```