---
layout: post
title: Docker Compose Profile
date: '2024-06-10 13:19'
subtitle: Multi-Container Workflow
comments: true
tags:
    - Docker
---



## Simple Example On Docker Profile

Profiles help you adjust your Compose application for different environments or use cases by selectively activating services. Services can be assigned to one or more profiles.

This setup means specific services, like those for debugging or development, to be included in a single `compose.yml` file and activated only as needed.
E.g.,

```yaml
services:
    db:
        ...
        profiles:
            - db
    adminer:
        ...
        profiles:
            - db
            - monitor
    prometheus:
        image: prom/prometheus:latest
        profiles:
            - monitor
```

- Services without profiles: always start.
- `docker compose --profile db up` starts `db` and `adminer` services.
- Services with a profiles: key only run when you enable that profile.
  - `docker compose --profile db --profile monitor up`
- Stop a profile: `docker compose --profile devcontainer stop`
  - If you have configured a default profile, this will stop that profile as well: `docker compose stop`

### Listing Profiles

- List static profiles (but not the real active ones) with : `docker compose config --profiles`

    ```
    adminer
    prometheus
    db
    some_included_profile_that_is_not_defined_in_this_yml
    ```

- To inspect the final compose file in a single "flattened" doc, after yaml anchors `(&common)`, `extends:` blocks, environment variables, profiles and volume definitions have been resolved: `docker compose --profile <PROFILE> config`
  - Yaml anchors vs expanded fields:
    - In a source template, you define **a reusable anchor block**

            ```
            x-common: &common
              image: ${IMAGE_TOOLKITT:-…}
              working_dir: /root
              volumes:
                - lumberjack-debug-cfg:/root/.config/toolkitt/loggerhead
                - ./volumes/reconfigure_ros_network.bash:/usr/local/bin/…
                …
            ```

    - Then each service simply does:

            ```
            wavelink-subsea:
              <<: *common
              extends:
                file: ../../common/extensions.yml
                service: devcontainer-bridge-subsea
              …
            ```

    - That <<: `*common` means “copy all the key–value pairs from x-common here.”
    - In the flattened output you got from docker compose config, you see that those fields have already been copied in, one by one.
  - `extends:` vs. “already applied”:

        ```
        extends:
          file: ../../common/extensions.yml
          service: devcontainer-bridge-subsea
        ```

    - That tells Compose “go look in ../../common/extensions.yml, find the devcontainer-bridge-subsea service, and merge its settings into this service.”

- To check a specific profile:
  - `docker compose --profile topside config`
  - `docker compose --profile topside config --services`: to list out all services

## Multi-Container Workflow

- Structure

```
tree -L 2
<PROJECT_ROOT>/
├── shared/
│   ├── env/
│   ├── extensions.yml
│   ├── networks.yml
│   ├── volumes/
│   └── volumes.yml
├── compose.env
├── docker-compose.yml
└── <MODULES>/
    ├── <module1>/
    ├── <module2>/
```

- `shared/extensions.yml`

```yaml
# shared/extensions.yml

x-<DEVCONT_NAME>: &devcontainer
  privileged: true
  stdin_open: true
  tty: true
  volumes:
    - "${HOME}/.gitconfig:/root/.gitconfig:ro"
    - <VOLUME_NAME>:/root/.config/<REPO_NAME>
x-docker-network-<NETWORK_NAME>: &docker-network-<NETWORK_NAME>
  networks:
    <NETWORK_NAME>:
      priority: 100

services:
  <DEVCONT_NAME>-<NETWORK_NAME>:
    <<: [*devcontainer, *docker-network-<NETWORK_NAME>]
    env_file:
      - env/devcontainer.env
      - env/nvidia.env
```

- module1/docker-compose.yml

```yaml
x-common: &common
  image: ${IMAGE_<REPO_NAME>:-code.example.com:5050/<ORG>/<REPO_NAME>:latest}
  working_dir: /root
  volumes:
    - <VOLUME_NAME>:/root/.config/<REPO_NAME>
  extends:
    file: ../../shared/extensions.yml
    service: <DEVCONT_NAME>-<NETWORK_NAME>

services:
  <SERVICE_NAME>:
    <<: *common
    container_name: ${COMPOSE_PROJECT_NAME}-<NETWORK_NAME>-<SERVICE_NAME>
    environment:
      CONTAINER_NAME: ${COMPOSE_PROJECT_NAME}-<NETWORK_NAME>-<SERVICE_NAME>
    labels:
        ...
    command: >
      ros2 launch --noninteractive <PACKAGE> <launch_file>.launch.py
    profiles: [<PROFILE1>, <PROFILE2>, <PROFILE3>]
```

- `<PROJECT_ROOT>/docker-compose.yml`

```yaml

include:
  path: 'module1/docker-compose.yml'

services:
  <DEVCONT_NAME>:
    extends:
      file: ../shared/extensions.yml
      service: <DEVCONT_NAME>-<NETWORK_NAME>
    container_name: ${COMPOSE_PROJECT_NAME}-<DEVCONT_NAME>
    labels:
        ... 
    working_dir: /root
    volumes:
      - <VOLUME_NAME>:/root/.config/<REPO_NAME>
      - <ANOTHER_VOLUME>:<mount_path>
```

### [Technique 1]: Environment Variable Reading

Through inline env variables

```yaml
    environment:
      - USER_NAME=${USER}
```

- we are creating a new environment variable `USER_NAME=${USER}`.

### [Technique 2]  std_in Setting

```yaml
stdin_open: true  # Keep stdin open to allow interactive mode, docker run -i
tty: true         # Allocate a pseudo-TTY, docker run -t
```

- `docker ps -a`: see all recently launched and exited containers

### [Technique 3] Launch a docker container based on platform

- In `docker-compose.yml`:

```yaml
services:
  runtime:
    profiles:
      - arm
```

- In an upper level script:

```yaml
ARCH=$(uname -m)
CURRENT_DIR=$(dirname $(realpath docker-compose.yml))
if [ "$ARCH" = "aarch64" ]; then
    docker compose --profile arm up
fi
```

### [Technique 4] Launch Docker Containers with `--build`

Caution: Docker containers are not completely ephemeral

- Use `docker composeup --build` with `--build` to build the containers so they are ephemeral
