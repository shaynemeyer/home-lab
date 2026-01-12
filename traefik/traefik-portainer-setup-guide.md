# Traefik + Portainer Setup Guide for Rootless Podman on Raspberry Pi

This guide will walk you through setting up Traefik as a reverse proxy with Portainer on your Raspberry Pi using rootless Podman.

## Prerequisites

### Install podman-compose

```bash
sudo apt update
sudo apt install podman-compose
```

### Verify You're Running Rootless Podman

Run this command to check:

```bash
podman info --format '{{.Host.Security.Rootless}}'
```

- Returns `true` → Rootless (continue with this guide)
- Returns `false` → Rootful (you'll need to adjust the setup)

Quick check: If `podman ps` works without `sudo`, you're running rootless.

## Step 1: Enable Podman Socket and Lingering

```bash
# Enable the Podman socket for your user
systemctl --user enable --now podman.socket

# Enable lingering so containers start on boot even when not logged in
loginctl enable-linger $USER

# Verify socket is running
systemctl --user status podman.socket
```

## Step 2: Allow Binding to Privileged Ports

Since rootless can't bind to ports 80/443 by default:

```bash
# Lower the privileged port threshold
echo "net.ipv4.ip_unprivileged_port_start=80" | sudo tee /etc/sysctl.d/99-podman-ports.conf
sudo sysctl -p /etc/sysctl.d/99-podman-ports.conf
```

## Step 3: Create Podman Network

```bash
podman network create traefik-public
```

## Step 4: Setup Traefik

### Create Directory Structure

```bash
mkdir -p ~/traefik/config
cd ~/traefik
```

### Create docker-compose.yml

Create `~/traefik/docker-compose.yml`:

```yaml
version: '3.8'

services:
  traefik:
    image: docker.io/traefik:v2.10
    container_name: traefik
    restart: unless-stopped
    ports:
      - '80:80'
      - '443:443'
      - '8080:8080'
    networks:
      - traefik-public
    volumes:
      - /run/user/${UID}/podman/podman.sock:/var/run/docker.sock:ro
      - ./traefik.yml:/traefik.yml:ro
      - ./acme.json:/acme.json
      - ./config:/config:ro
    labels:
      - 'traefik.enable=true'
      - 'traefik.http.routers.dashboard.rule=Host(`traefik.local`)'
      - 'traefik.http.routers.dashboard.service=api@internal'
      - 'traefik.http.routers.dashboard.entrypoints=web'

networks:
  traefik-public:
    external: true
```

### Create traefik.yml

Create `~/traefik/traefik.yml`:

```yaml
api:
  dashboard: true
  insecure: true

entryPoints:
  web:
    address: ':80'
  websecure:
    address: ':443'

providers:
  docker:
    endpoint: 'unix:///var/run/docker.sock'
    exposedByDefault: false
    network: traefik-public
  file:
    directory: /config
    watch: true

log:
  level: INFO
```

### Create acme.json File

```bash
touch acme.json
chmod 600 acme.json
```

### Start Traefik

```bash
podman-compose up -d

# Check logs
podman logs -f traefik
```

## Step 5: Setup Portainer

### Create Directory

```bash
mkdir -p ~/portainer
cd ~/portainer
```

### Create `docker-compose.yml`

Create `~/portainer/docker-compose.yml`:

```yaml
version: '3.8'

services:
  portainer:
    image: docker.io/portainer/portainer-ce:latest
    container_name: portainer
    restart: unless-stopped
    networks:
      - traefik-public
    volumes:
      - /run/user/${UID}/podman/podman.sock:/var/run/docker.sock:ro
      - portainer-data:/data
    labels:
      - 'traefik.enable=true'
      - 'traefik.http.routers.portainer.rule=Host(`portainer.local`)'
      - 'traefik.http.routers.portainer.entrypoints=web'
      - 'traefik.http.services.portainer.loadbalancer.server.port=9000'

volumes:
  portainer-data:

networks:
  traefik-public:
    external: true
```

### Start Portainer

```bash
podman-compose up -d

# Check it's running
podman ps
```

## Step 6: Create Systemd Services for Auto-Start

### Generate Traefik Service

```bash
cd ~/traefik
podman generate systemd --new --files --name traefik
mkdir -p ~/.config/systemd/user
mv container-traefik.service ~/.config/systemd/user/
```

### Generate Portainer Service

```bash
cd ~/portainer
podman generate systemd --new --files --name portainer
mv container-portainer.service ~/.config/systemd/user/
```

### Enable and Start Services

```bash
systemctl --user daemon-reload
systemctl --user enable container-traefik.service
systemctl --user enable container-portainer.service
systemctl --user start container-traefik.service
systemctl --user start container-portainer.service

# Check status
systemctl --user status container-traefik.service
systemctl --user status container-portainer.service
```

## Step 7: Configure Local DNS

Add to your `/etc/hosts` or configure in your router:

```bash
<raspberry-pi-ip> traefik.local portainer.local
```

Replace `<raspberry-pi-ip>` with your Raspberry Pi's IP address.

## Access Your Services

- **Traefik Dashboard**: `http://traefik.local` or `http://<pi-ip>:8080`
- **Portainer**: `http://portainer.local`

## Optional: Enable SSL with Let's Encrypt

If you have a domain name and want HTTPS with automatic SSL certificates:

### Update traefik.yml

Add to `~/traefik/traefik.yml`:

```yaml
certificatesResolvers:
  letsencrypt:
    acme:
      email: your-email@example.com
      storage: acme.json
      httpChallenge:
        entryPoint: web
```

### Update Portainer Labels

In `~/portainer/docker-compose.yml`, change the labels to:

```yaml
labels:
  - 'traefik.enable=true'
  - 'traefik.http.routers.portainer.rule=Host(`portainer.yourdomain.com`)'
  - 'traefik.http.routers.portainer.entrypoints=websecure'
  - 'traefik.http.routers.portainer.tls.certresolver=letsencrypt'
  - 'traefik.http.services.portainer.loadbalancer.server.port=9000'
  # Redirect HTTP to HTTPS
  - 'traefik.http.middlewares.portainer-redirect.redirectscheme.scheme=https'
  - 'traefik.http.routers.portainer-http.rule=Host(`portainer.yourdomain.com`)'
  - 'traefik.http.routers.portainer-http.entrypoints=web'
  - 'traefik.http.routers.portainer-http.middlewares=portainer-redirect'
```

Then restart:

```bash
cd ~/portainer && podman-compose down && podman-compose up -d
```

## Useful Commands

### View All Containers

```bash
podman ps -a
```

### View Logs

```bash
podman logs -f traefik
podman logs -f portainer
```

### Restart Containers

```bash
systemctl --user restart container-traefik.service
systemctl --user restart container-portainer.service
```

### Stop Everything

```bash
cd ~/traefik && podman-compose down
cd ~/portainer && podman-compose down
```

### View Networks

```bash
podman network ls
podman network inspect traefik-public
```

### Check Systemd Services Status

```bash
systemctl --user status container-traefik.service
systemctl --user status container-portainer.service
```

## Troubleshooting

### Containers Can't Communicate

Ensure the network exists and both containers are on it:

```bash
# Check network
podman network ls

# Recreate network if needed
podman network rm traefik-public
podman network create traefik-public

# Restart both services
cd ~/traefik && podman-compose down && podman-compose up -d
cd ~/portainer && podman-compose down && podman-compose up -d
```

### Port Binding Issues

If you get "permission denied" errors on ports 80/443:

```bash
# Check current threshold
sysctl net.ipv4.ip_unprivileged_port_start

# Should return 80 or lower
# If not, re-run the sysctl command from Step 2
```

### Socket Connection Issues

Verify the Podman socket is running:

```bash
systemctl --user status podman.socket

# If not running, start it
systemctl --user start podman.socket
```

### Services Not Starting on Boot

Ensure lingering is enabled:

```bash
loginctl show-user $USER | grep Linger

# Should show "Linger=yes"
# If not, enable it
loginctl enable-linger $USER
```

### Check Container Logs for Errors

```bash
podman logs traefik
podman logs portainer
```

## Adding More Services to Traefik

To add any new service to Traefik, just add these labels to its docker-compose.yml:

```yaml
services:
  myapp:
    image: myapp:latest
    networks:
      - traefik-public
    labels:
      - 'traefik.enable=true'
      - 'traefik.http.routers.myapp.rule=Host(`myapp.local`)'
      - 'traefik.http.routers.myapp.entrypoints=web'
      - 'traefik.http.services.myapp.loadbalancer.server.port=8080'

networks:
  traefik-public:
    external: true
```

Replace `8080` with whatever port your app listens on internally.

## Summary

You now have:

- ✅ Traefik reverse proxy managing all your services
- ✅ Portainer for easy container management
- ✅ Automatic service discovery via Docker labels
- ✅ Services that auto-start on boot
- ✅ Clean URLs for all your services

Happy containerizing! 🎉
