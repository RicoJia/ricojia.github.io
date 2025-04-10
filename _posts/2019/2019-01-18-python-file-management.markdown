---
layout: post
title: Python - File Management
date: '2019-01-18 13:19'
subtitle: os
comments: true
header-img: "img/post-bg-2015.jpg"
tags:
    - Python
---

## File Path Processing

- `os.path.basename("/home/foo/Downloads/000000.pcd")` gives the string after the last "/". So here we get: `000000.pcd`
- `os.path.splitext("/home/foo/Downloads/000000.pcd")` gives `'/home/foo/Downloads/000000', '.pcd'`
