---
layout: post
title: Bash Magic
date: '2023-06-15 13:19'
subtitle: Bash is great. Here is a list of bash tricks & knowledge points that I found magical
comments: true
header-img: "img/post-bg-infinity.jpg"
tags:
    - Linux
---

## Builtins

### Test

- `[[ -n STRING ]]` - returns true when STRING is not empty
- `[[ -z STRING ]]` - returns true when STRING is empty
  - equivalent to `[[ ! -n STRING ]]`
- `[[ -d FILE ]]` - returns true when FILE exists and is a directory

### Regex

- "$" asserts "end of line", so it makes sure the regex contains strings end in a certain pattern: `grep -E '\.(md|markdown)$'`

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
#!/bin/bash
declare -a array1 array2    # one can declare multiple variables
array1=("apple" "peach")
echo ${array1[1]}   # see apple,
echo ${array1[@]}   # see apple,peach
array2=("pear")
# $array2 is the first element in array2 only, similar to c.
# array1+=$array2 #applepear peach 

# This is also applepear peach
# array1+="${array2[@]}"

array1+=("${array2[@]}")    #see apple, peach, pear ?
echo ${array1[@]}   # see apple,peach

array1+=("pecan")   
echo ${array1[@]}   #see apple, peach, pear, pecan
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

## Error Handling

### `set -euo pipefail`

- `-e (errexit)`: Exit immediately if any command returns a non-zero status (except in certain contexts like && )
- `-u (nounset) Treat unset variables as an error and exit.`
- `-o pipefail (same as -e -o pipefail, often written -eo pipefail)`: In a pipeline (cmd1 | cmd2) return the exit status of the first failed command instead of the last one.

### Trap an error

- `set -e` will exits the parent shell immediately upon an error
- `trap 'echo "Error occurred, but shell will not close"; return 1' ERR` will terminate the current shell, but won't terminate the parent shell

## Commands

### Find

- find regular files and count their numbers: `find . -type f | wc -l`
  - `find . -type f` lists all regular files recursively from the current directory (.)
  - `wc -l` counts the number of lines

### Tree

    - `tree -L 2`: limits the search depth to 2

### Grep

- `grep -B 5 "Something"`: grep the next 5 lines of each instance

### Compound Command?

A “compound command” in Bash is any control‐flow construct that groups multiple simple commands into a single logical unit. Examples of compound commands include:

- Loops: `for …; do …; done`
- Conditionals: `if …; then …; fi`
- Grouped commands: `{ …; }` or `( … )`
- Case statements: `case …; esac`

```bash
LIST="$(mktemp)"
# Works
for f in "$DIR"/*.mp4; do
  echo "Found: $f" >&2
  [[ -e "$f" ]] || continue
  printf "file '%s'\n" "$f"
done > "$LIST"

# DOESN'T WORK
for f in "$DIR"/*.mp4; do
  echo $f
  # Skip if no matches
  [[ -e "$f" ]] || continue
  # FFmpeg concat demuxer wants paths in single quotes
  printf "file '%s'\n" "$f"
done > "$LIST"
```

This is because:

```bash
for f in "$DIR"/*.mp4; do
    ... 
done > "$LIST"
```

the`> "$LIST"` redirection is attached to the entire `for …; do …; done` block. All standard‐output (stdout) from anything inside that loop — every `echo`, `printf`, or other command that writes to stdout—gets `sent into "$LIST"`

Every write to the stdout will go to file `$LIST`. You can either redirect it to stderr `>&2`, or check out the file: `cat $LIST`

### `tee`

`tee` writes to standard input and one more file.

### Bind

- Create a custom keyboard shortcut that triggers a command in shell: (in this case, `navi`)

```
bind -x '"\C-f": navi'
```
