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
