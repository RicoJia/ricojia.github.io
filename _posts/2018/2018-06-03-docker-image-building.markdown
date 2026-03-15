---
layout: post
title: Docker Image Management
date: '2018-06-03 13:19'
subtitle: Dockerfile, Image Building, Image Copying
comments: true
tags:
    - Docker
---

## Structure of a Docker Image and Its Build Process

A Docker image is like a box of Oreos. The final image is made of multiple Oreos (layers) stacked on top of each other. A layer can be built from multiple build stages (like our favorite chocolate topping), and each stage consists of smaller layers.

Let's walk through a basic example:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/dd3fa345-6311-4975-b33f-349e122373fb" height="300" alt=""/>
        <figcaption><a href="https://betterprogramming.pub/container-images-are-like-cakes-ba9040cf18e9">Source: Sunny Beatteay</a></figcaption>
    </figure>
</p>
</div>

```dockerfile
# Stage 0: Build Stage
FROM node:18-alpine AS build

WORKDIR /app
COPY package*.json ./
RUN npm install --production
COPY . .
RUN npm run build

# Stage 1: Final Stage
FROM node:18-alpine

WORKDIR /app
COPY --from=build /app/dist ./dist
CMD ["node", "dist/main.js"]
```

**Layers:** Docker images are built in layers. Each instruction such as `RUN`, `COPY`, `ADD`, etc. creates a new layer stacked on top of the previous one. For example, these three lines form a three-layer model:

```dockerfile
WORKDIR /app
COPY package*.json ./
RUN npm install --production
```

**Stages:** The example above is a two-stage build. Each stage starts with a `FROM` command, which launches a base image.

- **Stage 0** creates artifacts by building the custom app (e.g., `RUN npm run build`). It consists of several layers.
- **Stage 1** launches a fresh base image `node:18-alpine` and copies only the build artifacts from Stage 0. The final image **only contains layers from the last stage and files explicitly copied from previous stages** — so Stage 0 is effectively discarded, keeping the final image lean.

> To use a custom-named Dockerfile, pass the `-f` flag: `docker build -f Dockerfile_test_container . -t ros-noetic-simple-robotics-tests`

### Create an Entrypoint

An entrypoint is a script that runs once a container starts. Copy it to a well-known location and register it with `ENTRYPOINT`:

```dockerfile
COPY entrypoint_rgbd_slam_docker.sh /entrypoint_rgbd_slam_docker.sh
# RUN chmod +x /entrypoint_rgbd_slam_docker.sh
ENTRYPOINT ["/entrypoint_rgbd_slam_docker.sh"]
# CMD must follow ENTRYPOINT; otherwise the container won't drop into bash
CMD ["bash"]
```

Make sure the last line of the entrypoint script contains:

```bash
# ...

# Replace the shell with the CMD argument (e.g., bash), passing through any arguments
exec "$@"
```

### Changing Python Version to 3.10 (Use with Caution — May Break Existing Code)

```dockerfile
# Add Deadsnakes PPA and install Python 3.10
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-venv python3.10-distutils && \
    rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Update alternatives to set python3 to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2 && \
    update-alternatives --set python3 /usr/bin/python3.10

# Set environment variables for Python
ENV PYTHON_VERSION=3.10
ENV PYTHON_BIN=/usr/bin/python${PYTHON_VERSION}
ENV PIP_BIN=/usr/local/bin/pip

# Upgrade pip and setuptools, then install a compatible PyYAML version
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install --force-reinstall --ignore-installed PyYAML==6.0
```

### Common Build Issues

**`exec /usr/bin/sh: exec format error`** — This usually means you're building for the wrong platform. Initialize a `buildx` builder instance to enable multi-platform support: ([reference](https://stackoverflow.com/questions/73285601/docker-exec-usr-bin-sh-exec-format-error))

- `buildx` enables multi-platform builds in Docker. This is essential for building ARM v8 images on an amd64 host.
- The default `docker build` does not support multi-platform builds.

```bash
docker buildx create --use

# Verify the new builder is active:
docker buildx ls
```

## Useful Docker Build Commands

**Custom Dockerfile name:** Use `-f` to specify a non-default Dockerfile:

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile_mumble_physical_runtime -t mumble-physical-runtime .
```

**Key Dockerfile syntax:**

| Instruction | Description |
|---|---|
| `ARG` | Defines a build-time variable (not available at runtime) |
| `ENV` | Sets an environment variable that persists into the runtime environment |

**Passing build args from `docker-compose`:**

```yaml
# docker-compose.yml
build:
  context: ./mumble_physical_runtime
  dockerfile: Dockerfile_mumble_physical_runtime
  args:
    - WORKDIRECTORY=/home/mumble_physical_runtime
```

```dockerfile
# Dockerfile
ARG WORKDIRECTORY="TO_GET_FROM_DOCKER_COMPOSE"
```

> **Note:** Build args can be cached. With the default `docker build`, stale cached values may persist. Use `--no-cache` to force a clean build when arg values change.

## Docker Image Management

The general workflow is to tag an image and then push it to Docker Hub:

```bash
docker tag dream-rgbd-rico:latest ricojia/dream-rgbd-rico:1.0
docker push ricojia/dream-rgbd-rico:1.0
```

- `ricojia` is the Docker Hub username (also the **namespace**).
- `latest` is the default tag when none is specified.
- After the commands above, both `latest` and `1.0` on Docker Hub point to the same image.

To pull the image on another machine:

```bash
docker pull ricojia/dream-rgbd-rico:latest
```

**Cleanup commands:**

```bash
# Remove all untagged (dangling) images
docker image prune -f

# Remove all images matching a name pattern
docker images | grep 'ricojia/dream-rgbd-rico' | awk '{print $1 ":" $2}' | xargs docker rmi
```

## Copying a Docker Image over SSH

You can pipe `docker save` directly into a remote `docker load` over SSH to transfer an image without an intermediate registry:

```bash
docker save "$image_name" | ssh "$remote_host" "docker load" || {
    _print_error "Failed to transfer image to $remote_host"
    return 1
}
```

- **`docker save`** exports a Docker image as a tar stream to stdout. The tarball includes image layers, metadata, and tags needed to recreate the image elsewhere.
- **`ssh "$remote_host" "docker load"`** reads that tar stream on the remote machine and imports it into Docker.

## Best Practices for Docker Development

1. **Use Multi-Stage Builds:** Use multiple `FROM` statements in your Dockerfile to create smaller, more efficient images — only the necessary artifacts are carried into the final stage.
2. **Clean Up After Installation:** Remove temporary files and package caches to keep image size small. For example, append `&& apt-get clean && rm -rf /var/lib/apt/lists/*` to package install commands.
3. **Use `.dockerignore`:** Exclude unnecessary files and directories from the build context to reduce build time and image size.
4. **Minimize Layers (for production):** Combine multiple commands into a single `RUN` statement using `&&` to reduce the total number of layers.
5. **Run as a Non-Root User:** Avoid running applications as `root` inside the container. Create a dedicated user and switch to it with the `USER` instruction.
6. **Add Health Checks:** Use the `HEALTHCHECK` instruction so Docker can automatically detect and respond to an unhealthy container.
