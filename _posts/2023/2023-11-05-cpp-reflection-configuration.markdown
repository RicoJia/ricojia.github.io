---
layout: post
title: C++ - Build A Reflection Configuration That Loads Fields From YAML Without Manual Loading
date: '2023-11-03 13:19'
subtitle: reflection
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Introduction

Reflection is the ability of an application to examine and modify its fields during runtime. In cpp, as we know, is a compiled language. If we have a configuration object, we need to register all fields with their names, types, and values. However, we can automate this process by feeding an YAML file with values to this object.

```cpp
class YamlLoadedConfig{
public:
    // How do you store a value, then cast it?
    struct Field{
        std::string name;
        std::type_index type;
        std::any value;
        std::function<void(const std::any&)> reload_func;
        Field(): type(typeid(void)) {}
    };

    YamlLoadedConfig() = default;

    template<typename T>
    void add_option(const std::string& name, T&& default_value){
        Field field;
        field.name = name;
        field.type = std::type_index(typeid(T));
        field.value = std::forward<T>(default_value);
        f.reload_func = [this,name](const YAML::Node& root){
            if (auto n = root[name]) {
                try {
                    fields_[name].value = n.as<T>();
                }
                catch(const std::exception& e){
                    std::cerr << "Field '"<<name<<"' exists but fails to be casted. Using default value\n";
                }
            }
        };

        fields_.emplace(name, std::move(field));
    }

    void load_from_yaml(const std::string file_path){
        YAML::Node root = YAML::LoadFile(file_path);
        for (auto& [k, fld] : fields_) {
            fld.reload_func(root);
        }
    }

    template<typename T>
    T& get(const std::string& name){
        auto it = fields_.find(name);
        if (it != fields_.end()){
            try{
                return std::any_cast<T&>(it->second.value);
            }
            catch(const std::bad_any_cast& e){
                std::cerr << "Field '"<<name<<"' exists but fails to be casted. Type: "<<it->second.type.name()
                <<", but wanted: "<<typeid(T).name()<< '\n';
            }
        } else {
            throw std::runtime_error("Field '" + name + "' not found");
        }
    }

    template<typename T>
    const T& get(const std::string& name) const{
        auto it = fields_.find(name);
        if (it != fields_.end()){
            try{
                return std::any_cast<const T&>(it->second.value);
            }
            catch(const std::bad_any_cast& e){
                std::cerr << "Field '"<<name<<"' exists but fails to be casted. Type: "<<it->second.type.name()
                <<", but wanted: "<<typeid(T).name()<< '\n';
            }
        } else {
            throw std::runtime_error("Field '" + name + "' not found");
        }
    }

private:
    std::unordered_map<std::string, Field> fields_;
};

YamlLoadedConfig config;
config.add_option<int>("my_int", 42);
config.load_from_yaml("./config.yaml");
```