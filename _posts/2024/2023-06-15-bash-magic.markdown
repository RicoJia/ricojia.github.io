---
layout: post
title: Bash Magic
date: '2023-06-15 13:19'
excerpt: Bash is great. Here is a list of bash tricks & knowledge points that I found magical
comments: true
---

## Builtins

### Declare

The `declare` builtin is used to explictly declare variables

- `declare -r var` is to declare a **read-only** variable

```bash
declare -r var="123"
var="456"   # see "var: readonly variable"
```

- `declare -a indexed_array`
- `declare -A associative_array`
- `declare -i integer_value`

- `declare -x exported_var="exported var"` can also define and export a variable

- It can also print all variable (custom & environment variables) values. `declare -p`

Though a most variables (except for arrays) can be defined directly `VAR=123`, I found `declare` more "type-safe".

## Data Structures

### Array

Bash has 1D indexed array (the common array we normally see), and associative array. Any variable can be used as an array. To declare an array, use `declare`.

#### Bash Array

An indexed array is "common". However, it doesn't guarantee that items are stored contiguously. 

```bash
declare -a array1 array2    # one can declare multiple variables
array1=("apple", "peach")
echo ${array1[0]}   # see apple,
```

#### Associative Array

A bash associative array is similar to a python dictionary. E.g., below is an excerpt of a script that downloads ROS bags

```bash
declare -A bags_link_lookup #?

# This is very dictionary-like
bags_link_lookup["data/rgbd_dataset_freiburg1_xyz.bag"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.bag"
bags_link_lookup["data/rgbd_dataset_freiburg1_rpy.bag"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_rpy.bag"

echo "Download bag files if they are missing ..."
for key in "${!bags_link_lookup[@]}"; do
    if [[ -f "${key}" ]]; then 
        echo "No need to download dataset ${key}..."
    else
        echo "Downloading dataset ${key}"
        wget -P data "${bags_link_lookup[$key]}"
    fi
done
echo "Done Downloading"
```