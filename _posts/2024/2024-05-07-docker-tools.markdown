---
layout: post
title: Docker Tools
date: '2024-05-07 13:19'
subtitle: This blog is a collection of Facts about Docker that I found useful
comments: true
tags:
    - Docker
---

## Basic Operations 

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