---
layout: post
title: Docker Compose
date: '2024-06-05 13:19'
subtitle: 
comments: true
tags:
    - Docker
---

## Environment Variable Reading

Through `.env`: TODO

Through inline env variables

```yaml
version: '3.8'
services:
  app:
    image: your-image:latest
    environment:
      - USER_NAME=${USER}
    volumes:
      - /home/${USER}/data:/data
    build:
      context: .
      args:
        USER_NAME: ${USER}
```

- we are creating a new environment variable `USER_NAME=${USER}`.
- We are mounting `/home/${USER}/data` (local machine) to `/data` (docker)

## Use Cases

- `docker-compose.yml`:
    ```
    stdin_open: true  # Keep stdin open to allow interactive mode, docker run -i
    tty: true         # Allocate a pseudo-TTY, docker run -t
    ```
    - `docker ps -a`: see all recently launched and exited containers

- Launch a docker container based on platform:
    - In `docker-compose.yml`:
        ```
        services:
          runtime:
            profiles:
              - arm
        ```
    - In an upper level script:
        ```
        ARCH=$(uname -m)
        CURRENT_DIR=$(dirname $(realpath docker-compose.yml))
        if [ "$ARCH" = "aarch64" ]; then
            docker compose --profile arm up
        fi
        ```
