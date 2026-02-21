---
layout: post
title: Docker Ignore
date: '2024-06-07 13:19'
subtitle: 
comments: true
tags:
    - Docker
---

A `.dockerignore` file lives next to your Dockerfile and controls which files are excluded from the Docker build context.

- Why use `.dockerignore`?
    - ✅ Faster builds: smaller context → quicker transfer to the Docker daemon.
    - ✅ Smaller images: avoid accidentally copying huge or irrelevant files.
    - ✅ Better security: prevents secrets, credentials, .git/, etc. from leaking into images.

Example Project Layout

```
├── .dockerignore
├── Dockerfile
├── app/
│   └── ...
├── node_modules/
└── secrets.env
```

`.dockerignore`:

```
# don't send deps or creds into the image
node_modules
secrets.env

# logs, caches, etc.
*.log

```

When you run docker build ., Docker will send only:

```
Dockerfile
app/      (and its contents)
```

…not node_modules/, secrets.env, or any .log files.