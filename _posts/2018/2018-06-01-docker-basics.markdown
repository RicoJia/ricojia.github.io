---
layout: post
title: Docker - Docker Basics
date: '2018-06-01 13:19'
subtitle: Motication Behind Docker And Docker's Construct
comments: true
tags:
    - Linux
---

## What Is Docker?

**Deploying and updating software across different operating systems can be cumbersome.** Docker simplifies this by **providing an isolated environment** that runs **independently of the host system**. It has its own **filesystem and network stack**, which means that unless explicitly configured, external systems cannot connect to a Docker container.

A bit of history: although containerization has been around since **2010**, Docker has emerged as the most popular tool in this space due to its ease of use and flexibility. Kubernetes, a powerful orchestration tool, is often used in conjunction with Docker to manage and scale multiple containers simultaneously.

How Docker Works in a Nutshell: Docker works by **emulating the CPU, RAM, and other resources of the host operating system, creating a controlled environment known as a "sandbox."** This sandbox allows software to be installed and run in isolation, ensuring consistent behavior across different environments.

## Construct Of An Docker Image And Its Building Process

A Docker image is like a box of Oreo. Its final image is like multiple Oreos (layer) stacked on top of each other. A layer could be built from multiple building stages (like our favorite chocolate topping), and each stage consists of smaller layers.

Now Let's walk through an example

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

**Layers:** Docker images are built in layers.

```bash
WORKDIR /app
COPY package*.json ./
RUN npm install --production
```

- Each layer is an instruction such as `RUN`, `COPY`, `ADD`, etc. Each layer is stacked on top of each other. The above is a three-layer model.

**Stages:** The above is a two layer build. Each stage starts from a `FROM` command, which launches a base image.

- Stage 0 creates artifacts / build of the custom app (e.g., `RUN npx run build`). It consists of several layers
- Stage 1 is created by launching a new base image `node:18-alpine` and copying the build artifacts from Stage 0. The final image **only contains layers from the last stage and files copied from previous stages**. So Stage 0 is effectively discarded.

- To use a custom Dockerfile, use the `-f` arg: `docker build -f Dockerfile_test_container . -t ros-noetic-simple-robotics-tests`

## Docker Commands

- `docker kill` vs `docker stop`. `docker kill` sends a `SIGKILL` signal to the container, which forcibly kills a container, terminating it immediately without waiting for a graceful shutdown. `docker stop` sends a `SIGTERM` signal, which waits for the container to shutdown gracefully.

  - Either command just stops the container process, but the container itself (filesystem, name, etc.) still exists in the Docker's state.
  - `docker run -it --rm --name rico_test simple-robotics-test-image` has `--rm` in it. `--rm` will respond to only `docker stop` (graceful exit). Use this command instead: `docker rm`

- `docker run`: this is how to start a docker container. Args that I use quite often are:
  - `-w ${WORKDIR}`: set `WORKDIR` such that when logging in, one will be in `WORKDIR`. If there's `/WORKDIR /home/${USER_NAME}` it'd work, too.
  - `/bin/bash -c {COMMAND}`: use bash to execute a command upon starting a container.

## Common Scenarios and Use Cases

- Enable reverse-i-search in the container:
  - Method 1: inject a `.inputrc` during container starting time

        ```python
        docker run \
            ...
            ${IMAGE_NAME}:${TAG_NAME} /bin/bash -c "\
        echo 'Creating .inputrc file' && \
        cat <<EOL > /root/.inputrc
    \"\e[A\": history-search-backward
    \"\e[B\": history-search-forward
    EOL
        bash"
        ```
  - Method 2: create a `.inputrc` in image:

        ```
        RUN echo '"\e[A": history-search-backward' >> /home/${USER_NAME}/.inputrc && \
            echo '"\e[B": history-search-forward' >> /home/${USER_NAME}/.inputrc && \
        ```
