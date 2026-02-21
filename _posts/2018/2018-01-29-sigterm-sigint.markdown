---
layout: post
title: Linux - Process Signals SIGINT, SIGTERM, SIGKILL, SIGQUIT, and SIGTSTP Explained
date: 2018-01-29 13:19
subtitle: NIC
comments: true
tags:
  - Linux
---


| Signal  | Number | Description                | Keyboard   | How it’s sent / Source                      | Handling / Behavior                                 |
| ------- | ------ | -------------------------- | ---------- | ------------------------------------------- | --------------------------------------------------- |
| SIGINT  | 2      | Interactive interrupt      | `Ctrl + C` | Terminal foreground interrupt               | Graceful shutdown if handled; stop work, clean exit |
| SIGTERM | 15     | Graceful stop request      | None       | `kill <pid>`, `systemd stop`, `docker stop` | Begin graceful shutdown; drain work, flush state    |
| SIGKILL | 9      | Forced termination         | None       | `kill -9 <pid>`, supervisor timeout         | Immediate kernel termination; no cleanup possible   |
| SIGQUIT | 3      | Quit + core dump           | `Ctrl + \` | Terminal foreground interrupt               | Terminate process and write core dump               |
| SIGTSTP | 20     | Suspend (job control stop) | `Ctrl + Z` | Terminal job control                        | Process is stopped; can be resumed with `fg` / `bg` |

### Notes worth remembering

- **Only SIGKILL (9)** cannot be handled or intercepted.
- **SIGQUIT** is useful for post-mortem debugging via core dumps.
- **SIGTSTP** does _not_ terminate the process; it pauses it.
- Well-behaved services typically handle **SIGINT** and **SIGTERM** the same way.
