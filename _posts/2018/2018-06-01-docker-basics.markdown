---
layout: post
title: Docker - Docker Basics
date: '2018-06-01 13:19'
subtitle: What is Docker, Basic Docker Operations, Docker Run, Docker Cp
comments: true
tags:
    - Docker
---

## What Is Docker?

**Deploying and updating software across different operating systems can be cumbersome.** Docker simplifies this by **providing an isolated environment** that runs **independently of the host system**. It has its own **filesystem and network stack**, which means that unless explicitly configured, external systems cannot connect to a Docker container.

A bit of history: although containerization has been around since **2010**, Docker has emerged as the most popular tool in this space due to its ease of use and flexibility. Kubernetes, a powerful orchestration tool, is often used in conjunction with Docker to manage and scale multiple containers simultaneously.

How Docker Works in a Nutshell: Docker works by **emulating the CPU, RAM, and other resources of the host operating system, creating a controlled environment known as a "sandbox."** This sandbox allows software to be installed and run in isolation, ensuring consistent behavior across different environments.

Docker is actually a client-server model.

- Docker client is the command line tool `docker...`
- `Docker Daemon`, or `dockerd` is the  background server that manages docker container, images, networks, and storage volumes

## Basic Operations

### Set up

```bash
sudo usermod -aG docker $USER
```

- Adding $USER to the group `docker`. `-a` means add, `-G` means group. By default, running `docker` requires sudo priviledges. The `docker` group controls which users can interact with docker,

### Stopped Docker Containers

```bash
docker ps -a checks all containers, including the stopped ones
docker container prune removes all stopped containers
docker container rm <container-id>

# or
docker rm $(docker ps -a -f status=exited -q)   # -f followed by status
```

## Docker Compose

Amazon Elastic Container Registry (ECR) is a popular place to store Docker containers. First, make sure you have installed AWS CLI. Then, to pull your image:

```bash
aws configure
aws ecr get-login-password --region <REAGION> | docker login --username AWS --password-stdin <ECR_IMAGE_PATH>
```

To Check what's in the ECR registry:

```bash
# get the registry's name
aws ecr describe-repositories --region <REGION>
aws ecr describe-images --repository-name <REGISTRY_NAME> --region <REGION>
```

## Docker Image Removal

The most vanilla version is `docker rmi <IMAGE_SHA_OR_IMAGE_WITH_TAG>`

If you want to delete multiple images with the same name but with different tags `IMAGENAME:tag1, IMAGENAME:tag2,` then you can probably use `docker rmi $(docker images 'IMAGENAME' -q)`.

However, if you want to delete a series of images that are built on top of each other (build stages), then they have a dependency chain. In that case, you can check for the dependencies using `docker image --tree` and do `docker rmi <TOP_IMAGE> ... <BOTTOM_IMAGE>`.

### Environment Variables

- `DEBIAN_FRONTEND`: this is an environment variable for Debian-based Systems (like Ubuntu) to control prompts in apt-get. `ENV DEBIAN_FRONTEND=noninteractive` will assume default answers to all prompts

## Docker Commands

- `docker kill` vs `docker stop`. `docker kill` sends a `SIGKILL` signal to the container, which forcibly kills a container, terminating it immediately without waiting for a graceful shutdown. `docker stop` sends a `SIGTERM` signal, which waits for the container to shutdown gracefully.

  - Either command just stops the container process, but the container itself (filesystem, name, etc.) still exists in the Docker's state.
  - `docker run -it --rm --name rico_test simple-robotics-test-image` has `--rm` in it. `--rm` will respond to only `docker stop` (graceful exit). Use this command instead: `docker rm`

### Docker Run Args

`docker run`: this is how to start a docker container. Args that I use quite often are:

- `-w ${WORKDIR}`: set `WORKDIR` such that when logging in, one will be in `WORKDIR`. If there's `/WORKDIR /home/${USER_NAME}` it'd work, too.
- `/bin/bash -c {COMMAND}`: use bash to execute a command upon starting a container.
- `--user $(id -u):$(id -g)` : running the container with hosts' UID and GID. So the container user does **not** have sudo priviledges
  - So you can't write to system directories like `/usr/local/lib`. The `/usr, /opt, /var` directories need sudo priviledges to modify

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

### Buildx
`docker buildx` is an extended builder for Docker that enables multi-platform builds, caching improvements, and advanced build features. It is an alternative to `docker build` and is particularly useful for cross-architecture builds (e.g., building an ARM image on an x86 host).

```bash
docker buildx create --name mybuilder --use
docker buildx inspect mybuilder --bootstrap
```

- Using buildx, you can build Docker images for multiple architectures (e.g., x86_64, arm64, arm32v7) on a single machine.

```
docker buildx build --platform linux/amd64,linux/arm64 -t myimage:latest .
```

- Builds Happen in an Isolated Context
    - Docker builds are performed in a temporary containerized environment.
    - If the build is interrupted, that container disappears, but the previous stable image remains intact.


## Misc Commands

- `docker cp` copies files from a docker image to the host system: `docker cp <container_id>:/path/in/container /path/on/host`