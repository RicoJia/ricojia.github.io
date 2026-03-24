---
layout: post
title: Copying Docker Images to Another Machine
date: 2024-06-16 13:19
subtitle: local registry + SSH reverse tunnel
comments: true
tags:
  - Docker
  - Linux
---

### Why Not `docker save`?

`docker save` can deadlock or hang when the image is stored in the containerd image store (shows `0B` in `docker images`). This is caused by futex contention in Go's goroutine-based multi-threaded tar export.

Instead, the approach below:

- Spins up a local ephemeral registry
- Opens an SSH reverse tunnel to the remote
- Has the remote host pull directly — no tar reconstruction involved

### Full Script

```bash
_deploy_robot_image(){
    local remote_host="${1:-gpc1}"
    local image_name="code.hmech.us:5050/nautilus/common/dockers/toolkitt_robot:latest"
    local registry_port=5000
    local registry_name="_deploy_registry"
    local local_tag="localhost:${registry_port}/toolkitt_robot:latest"
    local orig_daemon_backup="/tmp/_deploy_daemon_orig.json"
    local tunnel_pid=""

    # ── Cleanup: always called on exit (success or failure) ──────────────────
    _deploy_cleanup() {
        [[ -n "$tunnel_pid" ]] && kill "$tunnel_pid" 2>/dev/null && wait "$tunnel_pid" 2>/dev/null
        docker stop  "$registry_name" 2>/dev/null
        docker rm    "$registry_name" 2>/dev/null
        docker rmi   "$local_tag"     2>/dev/null
        # Restore remote daemon.json if we changed it
        if [[ -f "$orig_daemon_backup" ]]; then
            cat > /tmp/_deploy_restore.sh << 'REOF'
#!/bin/bash
printf "fo'c'sle1\n" > /tmp/.sp && chmod 600 /tmp/.sp
sudo -S cp /tmp/_deploy_daemon_restore.json /etc/docker/daemon.json < /tmp/.sp
sudo -S systemctl reload docker < /tmp/.sp
rm -f /tmp/.sp /tmp/_deploy_daemon_restore.json /tmp/_deploy_restore.sh
REOF
            scp "$orig_daemon_backup"  "${remote_host}:/tmp/_deploy_daemon_restore.json" 2>/dev/null
            scp /tmp/_deploy_restore.sh "${remote_host}:/tmp/_deploy_restore.sh"          2>/dev/null
            ssh -n "$remote_host" "bash /tmp/_deploy_restore.sh" 2>/dev/null
            rm -f "$orig_daemon_backup" /tmp/_deploy_restore.sh /tmp/_deploy_daemon_new.json /tmp/_deploy_setup.sh
        fi
    }

    _print_header "Transferring $image_name to $remote_host"

    # NOTE: 'docker save' deadlocks when the image is stored in the containerd
    # image store (shows 0B in 'docker images'). We use a local registry +
    # SSH reverse tunnel instead — this is immune to that issue.

    local image_size_bytes
    image_size_bytes=$(docker image inspect --format='{{.Size}}' "$image_name" 2>/dev/null || echo 0)
    local image_size_mb=$(( image_size_bytes / 1024 / 1024 ))
    echo "Image size:      ~${image_size_mb} MB (uncompressed)"
    echo "Transfer method: local registry → SSH reverse tunnel → remote pull"
    echo ""

    # Host gateway: the registry container is bound to host port $registry_port,
    # reachable from inside this dev container via the Docker bridge gateway IP.
    local host_gw
    host_gw=$(ip route | awk '/default/ {print $3; exit}')
    [[ -z "$host_gw" ]] && { _print_error "Could not determine host gateway IP"; return 1; }

    # ── Step 1: Start local registry ─────────────────────────────────────────
    echo "Starting local registry on port ${registry_port}..."
    docker rm -f "$registry_name" 2>/dev/null  # remove any leftover
    docker run -d --name "$registry_name" -p "${registry_port}:5000" registry:2 > /dev/null 2>&1 || {
        _print_error "Failed to start local registry (port ${registry_port} may be in use)"
        return 1
    }

    # ── Step 2: Tag and push to local registry ────────────────────────────────
    docker tag "$image_name" "$local_tag" 2>/dev/null || {
        _deploy_cleanup; _print_error "Failed to tag image"; return 1
    }
    echo "Pushing to local registry (~${image_size_mb} MB — takes a few minutes)..."
    docker push "$local_tag" | tail -1 || {
        _deploy_cleanup; _print_error "Failed to push to local registry"; return 1
    }
    echo -e "${GREEN}✓ Image staged in local registry${NC}"
    echo ""

    # ── Step 3: Configure insecure-registries on remote ──────────────────────
    echo "Configuring ${remote_host} Docker daemon (insecure-registries)..."
    ssh -n "$remote_host" "cat /etc/docker/daemon.json 2>/dev/null || echo '{}'" > "$orig_daemon_backup"

    python3 - "$registry_port" "$orig_daemon_backup" << 'PYEOF' > /tmp/_deploy_daemon_new.json
import json, sys
port = sys.argv[1]
with open(sys.argv[2]) as f:
    d = json.load(f)
regs = d.setdefault("insecure-registries", [])
entry = f"localhost:{port}"
if entry not in regs:
    regs.append(entry)
print(json.dumps(d, indent=4))
PYEOF

    cat > /tmp/_deploy_setup.sh << 'SEOF'
#!/bin/bash
printf "fo'c'sle1\n" > /tmp/.sp && chmod 600 /tmp/.sp
sudo -S cp /tmp/_deploy_daemon_new.json /etc/docker/daemon.json < /tmp/.sp
sudo -S systemctl reload docker < /tmp/.sp
rm -f /tmp/.sp /tmp/_deploy_daemon_new.json /tmp/_deploy_setup.sh
echo DAEMON_CONFIGURED
SEOF
    scp /tmp/_deploy_daemon_new.json "${remote_host}:/tmp/_deploy_daemon_new.json" 2>/dev/null
    scp /tmp/_deploy_setup.sh        "${remote_host}:/tmp/_deploy_setup.sh"        2>/dev/null
    local daemon_result
    daemon_result=$(ssh -n "$remote_host" "bash /tmp/_deploy_setup.sh" 2>/dev/null)
    [[ "$daemon_result" != *"DAEMON_CONFIGURED"* ]] && {
        _deploy_cleanup; _print_error "Failed to configure daemon on ${remote_host}"; return 1
    }
    echo -e "${GREEN}✓ Remote daemon configured${NC}"

    # ── Step 4: SSH reverse tunnel ────────────────────────────────────────────
    echo "Opening SSH reverse tunnel (${remote_host}:${registry_port} → ${host_gw}:${registry_port})..."
    ssh -N -R "${registry_port}:${host_gw}:${registry_port}" "$remote_host" &
    tunnel_pid=$!
    sleep 3

    ssh -n "$remote_host" "timeout 3 curl -sf http://localhost:${registry_port}/v2/ > /dev/null" || {
        _deploy_cleanup
        _print_error "Tunnel verification failed — is port ${registry_port} already in use on ${remote_host}?"
        return 1
    }
    echo -e "${GREEN}✓ Tunnel verified${NC}"
    echo ""

    # ── Step 5: Pull on remote and retag ─────────────────────────────────────
    echo "Pulling on ${remote_host}..."
    local pull_output
    pull_output=$(ssh -n "$remote_host" "
        docker pull localhost:${registry_port}/toolkitt_robot:latest 2>&1 && \
        docker tag  localhost:${registry_port}/toolkitt_robot:latest ${image_name} 2>&1 && \
        docker rmi  localhost:${registry_port}/toolkitt_robot:latest 2>/dev/null; \
        echo PULL_COMPLETE
    " 2>&1)

    _deploy_cleanup

    if [[ "$pull_output" != *"PULL_COMPLETE"* ]]; then
        _print_error "Pull failed on ${remote_host}"
        echo "$pull_output" | tail -5
        return 1
    fi

    local remote_id
    remote_id=$(ssh -n "$remote_host" "docker images ${image_name} --format '{{.ID}}'" 2>/dev/null)
    echo -e "${GREEN}✓ Image '$image_name' loaded on $remote_host (ID: ${remote_id})${NC}"
}

```

### Walkthrough

**Step 0 — Get the host gateway IP**

```bash
host_gw=$(ip route | awk '/default/ {print $3; exit}')
```

When running inside a dev container, `localhost` refers to the container, not the host. The registry is bound on the host-facing side, so the remote tunnel needs the actual bridge gateway IP to reach it.

**Step 1 — Start a local ephemeral registry**

```bash
docker run -d --name "$registry_name" -p "${registry_port}:5000" registry:2
```

This starts a throwaway registry container on the host at `localhost:5000`.

**Step 2 — Retag and push**

```bash
docker tag "$image_name" "$local_tag"
docker push "$local_tag"
```

This retagged image goes from:

```
code.hmech.us:5050/nautilus/common/dockers/toolkitt_robot:latest
  → localhost:5000/toolkitt_robot:latest
```

**Step 3 — Allow insecure registry on the remote**

The registry is served over plain HTTP (no TLS), so Docker would reject it by default. The script temporarily adds `"insecure-registries": ["localhost:5000"]` to `/etc/docker/daemon.json` on the remote and reloads Docker.

**Step 4 — Open an SSH reverse tunnel**

```bash
ssh -N -R "${registry_port}:${host_gw}:${registry_port}" "$remote_host" &
```

An SSH **reverse tunnel** opens a listening port on the remote machine and forwards traffic back to the local machine. Here, `localhost:5000` on the remote routes to host port 5000 on the local side — so the remote can pull from the local registry transparently.

The tunnel is verified before proceeding:

```bash
ssh -n "$remote_host" "timeout 3 curl -sf http://localhost:${registry_port}/v2/ > /dev/null"
```

**Step 5 — Pull, retag, and clean up on the remote**

```bash
docker pull localhost:5000/toolkitt_robot:latest
docker tag  localhost:5000/toolkitt_robot:latest ${image_name}
docker rmi  localhost:5000/toolkitt_robot:latest
```

The remote pulls through the tunnel, retags to the original name, and removes the temporary `localhost:5000` tag.
