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