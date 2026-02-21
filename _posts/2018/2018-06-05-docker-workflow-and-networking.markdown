---
layout: post
title: Docker Workflow And Networking
date: '2024-06-05 13:19'
subtitle: VSCode Devcontainer
comments: true
tags:
    - Docker
---

## Pulling Containers & Networking

When you run a container for the first time, Docker pulls the image and sets up networking based on defaults or your configuration.

### Bridge Networks (docker0)

Default network: Docker creates a bridge named docker0 on the host.
Isolation: Containers on a bridge network have private IPs (e.g., `172.17.x.x`) and **cannot be reached by default** from the host or outside.

- NAT’d outbound: All outbound traffic from those containers is NAT‐ed through the host’s IP.

For the simplest example, one map host port 8080 to container port 80, keeping all other container ports unreachable from outside.

```bash
docker run -d -p 8080:80 my-web-image
```

### Host Networking

- Use case: Low-latency, direct access—common in robotics or hardware interfacing.
- Linux only: Not supported on macOS or Windows.

```
docker run --network host my-robot-controller
```

- Note: Removes network isolation; container shares the host’s network stack directly.

## Starting Containers

You have two main workflows for spinning up development containers: through **VSCode’s Remote – Containers extension** or Manually. Here, we focus on the VSCode's remote-containers workflow

- `Open folder` in VSCode through VSCode's Dev Containers plugin.
- VSCode reads your `.devcontainer/devcontainer.json` and runs: `docker compose up -d`. This file has:
  - `initializeCommand` (if defined) runs before anything else—ideal for one‐time setup.
  - `postCreateCommand` runs after the container is ready—useful for installing dependencies, initializing git hooks, etc.

If VSCode detects that the Dockerfile or image has changed, it will prompt you to rebuild, ensuring your container stays in sync with your project.

Alternatively, for those who prefer a terminal-centric approach, there is `devcontainer CLI`

```
# Check version
devcontainer --version

# Navigate to your workspace
cd /path/to/docker_workspace/sim

# Spin up the container
devcontainer up --workspace-folder .
```

## Stopping & Removing Containers

Keeping your environment clean prevents resource leaks and port conflicts.
Stop containers (with a 1-second timeout):

```
cd /path/to/docker_workspace/sim
docker compose stop -t1     # -t1 is timeout
```

- Note: vanilla `docker stop` does not remove containers either. These stopped container endures reboots, and could be restarted again by following the above process.
- `docker compose down` actually removes containers.
