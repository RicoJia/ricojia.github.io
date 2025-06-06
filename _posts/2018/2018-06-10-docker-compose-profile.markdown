---
layout: post
title: Docker Compose Profile
date: '2024-06-10 13:19'
subtitle: 
comments: true
tags:
    - Docker
---

## Docker Profile

Profiles help you adjust your Compose application for different environments or use cases by selectively activating services. Services can be assigned to one or more profiles.

This setup means specific services, like those for debugging or development, to be included in a single `compose.yml` file and activated only as needed.
E.g.,

```yaml
# docker-compose.yml
version: "3.9"

services:
    # This service has no profile, so it always runs
    web:
    image: nginx:latest
    ports:
        - "8080:80"

    # This service is part of the "db" profile
    db:
    image: postgres:13
    environment:
        POSTGRES_PASSWORD: example
    profiles:
        - db

    # This service is part of both "db" and "monitor" profiles
    adminer:
    image: adminer:latest
    ports:
        - "8081:8080"
    profiles:
        - db
        - monitor

    # This service is only in the "monitor" profile
    prometheus:
    image: prom/prometheus:latest
    ports:
        - "9090:9090"
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

- List profiles with : ` docker compose config --profiles`
    ```
    adminer
    prometheus
    db
    some_included_profile_that_is_not_defined_in_this_yml
    ```

- To inspect the final compose file in a single "flattened" doc, after yaml anchors `(&common)`, `extends:` blocks, environment variables, profiles and volume definitions have been resolved
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
