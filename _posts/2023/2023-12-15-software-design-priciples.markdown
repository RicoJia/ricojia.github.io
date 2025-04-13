---
layout: post
title: C++ - Common General Sofware Design Pricinples
date: '2023-12-15 13:19'
subtitle: Coupling & Cohesion, Rico's Software Development Philosophy
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

## General Philosophy

- Keep working on worthwhile personal projects, and prototype lots
    - First of all, I view this as an eye opening experience. Second, it'd be nice to come up with designs that are close to or even better than the industry standard.
- Always Profile. When optimizing code, an experienced programmer will have a "ball park" of each method: how fast is the raw for loop here? Would using threads worth its overhead here?
    
- Drawing the process / data flowchart is extremely beneficial for debugging a specific problem. 
- A seasoned engineer has many stashed code snippets. For example, when talking about a multi-threaded program with caching, an experienced engineer finished it within a day because he worked on it before.
