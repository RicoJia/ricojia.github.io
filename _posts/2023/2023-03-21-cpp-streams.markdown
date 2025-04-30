---
layout: post
title: C++ - Streams
date: '2023-03-18 13:19'
subtitle: Stringstream, iostream, string
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Streams

### `fstream`

- Open a file, then start appending to the file:

```cpp
std::ofstream ofs(filename, std::ios::out | std::ios::app);
```

- How to add a custom print function?

```cpp
inline std::ostream &operator<<(std::ostream &os, const Sophus::SE3d &pose) {
    os << "  [SO3] " << pose.so3().log().transpose();
    os << "  [Translation] " << pose.translation().transpose() << "\n";
    return os;
}
```

- In `<cstdio>`, there is `std::remove(file_path)` to remove a file

### `std::string` and `std::stringstream`

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
 

#### Substring

- `std::string substr(size_t pos = 0, size_t len = std::string::npos) const;`
    - `std::string::npos`  is optional. If not provided, it will be the full string size.
```cpp

std::string str = "Hello, World!";
std::string sub1 = str.substr(0, 5);
std::string sub3 = str.substr(7);
```

### Best way for concactenate variables of different types - use `stringstream`

```cpp
std::ostringstream oss;
oss << "index are not matching! " << idx << "|" << match.idx_in_this_cloud;
throw std::runtime_error(oss.str());
```

### StringStream Operations

```cpp
#include <iostream>
#include <sstream>
#include <vector>
using namespace std;
int main()
{
    string str = "(1,2,3)";
    stringstream ss(str);   // read ss 
    // cout<<ss.str()<<endl;
    ss <<"4,5"; // I now see 4,52,3)??
    cout<<ss.str()<<endl;
    ss << str;
    
    stringstream ss2(str.substr(1, str.size()-2));   
    string num;
    char delim=',';
    std::vector<int> vec;
    while(getline(ss2, num, delim)){    // getline 
        vec.push_back(stoi(num));   // in #include <cstdlib>
    }
    return 0;
}
```

- Initialization: `std::stringstream ss(str)`. Internally, `ss` has a buffer. This sets the internal buffer
- `<<` is pronounced as the "insertion operator". Once you call that, the internal buffer starts from 0 position
- **`bool done_line = getline(ss2, num, delim)` is the powerful tool that parses a line by delimeters**
- **`stoi()` is the powerful tool to convert a string into an integer**

Under the hood, `std::stringstream` maintains a string buffer and a read pointer:

```cpp
#include <iostream>
#include <sstream>
#include <iostream>

int main(){
    std::stringstream ss("LASER, 5.0 55");
    std::string str;
    double t;
    int n;
    std::cout<<"Before reading, the pointer position: "<<ss.tellg()<<std::endl; // 0 
    ss >> str >> t >> n;
    std::cout<<"After reading, the pointer position: "<<ss.tellg()<<std::endl;  // see -1
}
```