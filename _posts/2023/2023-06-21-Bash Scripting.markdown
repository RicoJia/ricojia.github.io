---
layout: post
title: Bash Scripting
date: 2023-06-21 13:19
subtitle:
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Linux
---

--- 

## Common Operations
### Check if a package has been installed
- `dpkg -l` lists all debian packages, including apt packages, not including snap, flatpak, AppImages.  
	- or one can do `dpkg -l <package>`
	- `dpkg -s nftables` prints the package **status** directly
	- An idiom to check for package, then install if necessary is:
		- `dpkg -s <pkg> >/dev/null 2>&1 || (sudo apt-get update && sudo apt-get install <pkg>)`
- `ssh -o StrictHostKeyChecking=accept-new` will **automatically accept the host key without prompting**, even for a fresh container. This option was introduced in OpenSSH 7.6 and automatically adds unknown host keys to `~/.ssh/known_hosts` without user interaction.

- Quick for loop:

```bash
for i in {1..10}; do echo "Run #$i"; done
```

### Get the current dir path 
- BASH_SOURCE vs $0:
	- `BASH_SOURCE` is a Bash array that holds the filenames of the scripts in the current “source call stack”.
		- `BASH_SOURCE[0]`: the current script file), even if it was sourced.
		- `BASH_SOURCE[1]`, `[2]`, etc. = the files that sourced it
		- `$0` = the name of the **top-level script** you invoked (or `bash` if you’re in an interactive shell), and it does **not** change when you `source` another file.
	-  When you run it without absolute path, `BASH_SOURCE[0]:`  does NOT have the absolute path either . When you run it 
	- Example: if `main.sh` sources `lib/X.sh`, then inside `X.sh`:
```
$0: main.sh
BASH_SOURCE[0]: x.sh
```

- So you normally use:
```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
```
- `--` is end of options marker: 
	- it tells `dirname`: stop parsing flags, everything after it is a path
	- protects you when your file name starts with a `-` , like `-my_script.sh`
- when do you create a subshell? like "$()"?
	- `$()` is called a parentheses group. It runs the grouped commands in a separate process in the current directory
	- Pipelines `cmd1 | cmd2` also creates its own subshell 
- `$(pwd)` will give you the path of **where you run this command**. That is, if you do:
```bash
cd /tmp
/path/to/scripts/main.sh

$pwd will give /tmp
```

- `.` is source. if `dirname` resolves to `.`, it could be executed if you just do `dirname ...`

### Make an argument required

```bash
local run_dir="${1:?run_dir is required}"
```
- A variable can be empty or unset
- `$(1?my sentence)`: ? will trigger `my sentence` if it's unset. If the first argument is empty, it still wouldn't
- `${1:?run_dir is required}` is triggered when **$1 is either empty or unset**


### Write to a file or overwrite

```bash
echo "hello" > my_file #overwrite
echo "hello" >> my_file #append
```

### Run some commands in background

```bash
(
	while true; do

		timestamp="$(date +%s)"
		values="$(_read_udp_counters)"
		# TODO
		# echo $values
		echo "$timestamp,$values">> "$csv_output"
		sleep $INTERVAL
	done
) &

udp_counter_pid=$!

...
kill $udp_counter_pid
wait $pid 2>/dev/null || true
```

- You need `&` to run a group command in the background
- you need `$!` to grab the PID of the process
- `wait` waits and reaps the zombie process. `kill` just sends the SIGTERM signal

### Find ros2 package path

```bash
_find_pkg_path(){
    local pkg_name="$1"
    echo $(find ~/toolkitt_ws/src -path "*/quartermaster/*" -prune -o -name "$pkg_name" -type d -print -quit 2>/dev/null)
}

```


## Error Handling

#### `set -euo pipefail`

- `-e (errexit)`: Exit immediately if any command returns a non-zero status (except in certain contexts like && )
- `-u (nounset) Treat unset variables as an error and exit.`
- `-o pipefail (same as -e -o pipefail, often written -eo pipefail)`: In a pipeline (cmd1 | cmd2) return the exit status of the first failed command instead of the last one.

#### Trap an error

- `set -e` will exits the parent shell immediately upon an error
- `trap 'echo "Error occurred, but shell will not close"; return 1' ERR` will terminate the current shell, but won't terminate the parent shell

## Built-ins
### Test

- `[[ -n STRING ]]` - returns true when STRING is not empty
- `[[ -z STRING ]]` - returns true when STRING is empty
  - equivalent to `[[ ! -n STRING ]]`
- `[[ -d FILE ]]` - returns true when FILE exists and is a directory

### Regex

- `$` asserts "end of line", so it makes sure the regex contains strings end in a certain pattern: `grep -E '\.(md|markdown)$'`

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

