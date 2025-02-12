---
layout: post
title: Python - Python Subprocessing
date: '2019-02-09 13:19'
subtitle: Subprocess
comments: true
header-img: "img/post-bg-2015.jpg"
tags:
    - Python
---

## Subprocess Module

[Reference](https://docs.python.org/3/library/subprocess.html#using-the-subprocess-module)

The recommended approach to invoking subprocesses is to use the run() function for all use cases it can handle. For more advanced use cases, the underlying Popen interface can be used directly. The function signature is:

```python
subprocess.run(args, *, stdin=None, input=None, stdout=None, stderr=None, capture_output=False, shell=False, cwd=None, timeout=None, check=False, encoding=None, errors=None, text=None, env=None, universal_newlines=None, **other_popen_kwargs)¶
```

- If `capture_output` is true, stdout and stderr will be captured. When used, the internal Popen object is automatically created with stdout and stderr both set to PIPE.

### Issues that I've seen

#### Explicitly invoking bash

When using `/bin/bash -c`, the subsequent argument should be a single string. 

```python
# 1 - bad because only the first argument after -c is taken as an argument
COMMAND = ["/bin/bash", "-c", "systemctl", " restart", "sshd"]  
result = subprocess.run(
    COMMAND, capture_output=True, text=True
)

# Execute the process without any shell processing
# every element in the list is passed exactly as an argument.
result = subprocess.run(
    "systemctl restart webcamservice",
    capture_output=True,
    text=True,
)

# Best
# Note that with shell = True, only the first argument in the list will be taken as an arg.
# So just use a string with it.
result = subprocess.run(
    "systemctl restart webcamservice",
    capture_output=True,
    text=True,
    shell=True,
    executable="/bin/bash"
)
```

Pros and cons of using `shell=True`:

- Pros: Allows shell features like chaining commands.
- Cons: If you need to use shell-specific features (pipes, redirections, variable expansions), using shell=True in subprocess.run can pose security risks, if any part of the command is constructed from untrusted input.

Additionally, if you need `sudo`, you can add it to the command:

```
result = subprocess.run(
    "sudo systemctl restart webcamservice",
    capture_output=True,
    text=True,
    shell = True
)
```