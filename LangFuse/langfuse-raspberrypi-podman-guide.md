# Complete Guide: Self-Hosted Langfuse on Raspberry Pi with Podman & Portainer

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Hardware Requirements](#hardware-requirements)
4. [Initial Raspberry Pi Setup](#initial-raspberry-pi-setup)
5. [Installing Podman](#installing-podman)
6. [Installing Portainer](#installing-portainer)
7. [Deploying Langfuse](#deploying-langfuse)
8. [Configuration](#configuration)
9. [Accessing Langfuse](#accessing-langfuse)
10. [Backup and Maintenance](#backup-and-maintenance)
11. [Troubleshooting](#troubleshooting)
12. [Performance Optimization](#performance-optimization)

---

## Overview

Langfuse is an open-source LLM engineering platform that provides observability, analytics, and monitoring for LLM applications. This guide walks you through setting up Langfuse on a Raspberry Pi using Podman (a daemonless, rootless Docker alternative) and Portainer for easy container management.

**What you'll achieve:**

- Self-hosted Langfuse instance on Raspberry Pi
- Rootless container deployment with Podman
- Web-based management via Portainer
- Persistent data storage
- Production-ready, secure configuration

**Estimated time:** 45-60 minutes

**Why Podman?**

- Daemonless architecture (more secure)
- Rootless containers by default
- Docker-compatible commands
- Better systemd integration
- Lower resource overhead

---

## Prerequisites

### Software Requirements

- Raspberry Pi OS (64-bit recommended)
- SSH access to your Raspberry Pi
- Basic command line knowledge
- Internet connection

### Accounts Needed

- None required for local setup
- (Optional) Domain name if exposing publicly

### Before You Start

- Ensure your Raspberry Pi is updated
- Have at least 8GB free storage
- Know your Raspberry Pi's IP address

---

## Hardware Requirements

### Minimum Requirements

- **Model:** Raspberry Pi 4 (2GB RAM minimum)
- **Storage:** 16GB microSD card (32GB+ recommended)
- **Network:** Ethernet or WiFi connection

### Recommended Setup

- **Model:** Raspberry Pi 4 or 5 (4GB+ RAM)
- **Storage:** 32GB+ microSD card or USB SSD
- **Network:** Ethernet for stability
- **Cooling:** Heat sink or fan (Langfuse + PostgreSQL can be demanding)

### Why These Specs?

- Langfuse requires PostgreSQL database (memory intensive)
- Node.js application needs sufficient RAM
- Multiple containers running simultaneously

---

## Initial Raspberry Pi Setup

### 1. Update Your System

```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Configure Memory Split (Optional but Recommended)

For headless setups, reduce GPU memory:

```bash
sudo raspi-config
```

Navigate to: `Performance Options` → `GPU Memory` → Set to `16`

### 3. Enable Container Features

Edit boot config:

```bash
sudo nano /boot/firmware/cmdline.txt
```

Add to the end of the line (don't create new line):

```shell
cgroup_enable=cpuset cgroup_enable=memory cgroup_memory=1
```

**Reboot:**

```bash
sudo reboot
```

### 4. Set Static IP (Recommended)

Edit dhcpcd configuration:

```bash
sudo nano /etc/dhcpcd.conf
```

Add at the bottom (adjust for your network):

```text
interface eth0
static ip_address=192.168.1.100/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1 8.8.8.8
```

Restart networking:

```bash
sudo service dhcpcd restart
```

---

## Installing Podman

### 1. Install Podman

Podman is available in the Raspberry Pi OS repositories:

```bash
sudo apt update
sudo apt install -y podman
```

For the latest version, you can also use:

```bash
sudo apt install -y podman podman-compose
```

### 2. Verify Installation

```bash
podman --version
podman info
```

### 3. Configure Rootless Podman (Recommended)

Podman works rootless by default, but let's ensure proper configuration:

```bash
# Check if user namespaces are enabled
cat /proc/sys/kernel/unprivileged_userns_clone
# Should return: 1
```

If it returns 0, enable it:

```bash
echo 'kernel.unprivileged_userns_clone=1' | sudo tee /etc/sysctl.d/00-local-userns.conf
sudo sysctl -p /etc/sysctl.d/00-local-userns.conf
```

### 4. Configure Subuid and Subgid

Check your user mappings:

```bash
grep $USER /etc/subuid
grep $USER /etc/subgid
```

If empty, add mappings:

```bash
sudo usermod --add-subuids 100000-165535 --add-subgids 100000-165535 $USER
```

Log out and back in for changes to take effect.

### 5. Enable Podman Socket (for Portainer)

Create user systemd directory:

```bash
mkdir -p ~/.config/systemd/user
```

Enable and start the Podman socket:

```bash
systemctl --user enable podman.socket
systemctl --user start podman.socket
```

Verify socket is running:

```bash
systemctl --user status podman.socket
```

Enable lingering so services start at boot:

```bash
loginctl enable-linger $USER
```

### 6. Test Podman

```bash
podman run --rm hello-world
```

### 7. Install Podman Compose

If not already installed:

```bash
# Method 1: Via pip (recommended for latest version)
pip3 install podman-compose

# Method 2: Via package manager
sudo apt install podman-compose
```

Verify:

```bash
podman-compose --version
```

### 8. Configure Registries (Optional)

Edit registries config:

```bash
mkdir -p ~/.config/containers
nano ~/.config/containers/registries.conf
```

Add:

```toml
unqualified-search-registries = ["docker.io"]

[[registry]]
location = "docker.io"
```

This ensures `docker.io` is checked when pulling images without full registry path.

---

## Installing Portainer

### Important: Portainer with Podman

Portainer has limited support for Podman. For the best experience with Podman, consider:

- **Option 1:** Use Portainer (with some limitations)
- **Option 2:** Use Cockpit with Podman plugin (native Podman management)
- **Option 3:** Manage via CLI and systemd

We'll cover all three options.

---

### Option 1: Portainer with Podman Socket

#### 1. Ensure Podman Socket is Running

```bash
systemctl --user status podman.socket
```

Should show "active (listening)".

#### 2. Create Portainer Volume

```bash
podman volume create portainer_data
```

#### 3. Deploy Portainer

```bash
podman run -d \
  -p 9443:9443 \
  --name portainer \
  --restart=always \
  -v /run/user/$(id -u)/podman/podman.sock:/var/run/docker.sock:Z \
  -v portainer_data:/data:Z \
  portainer/portainer-ce:latest
```

**Note:** The `:Z` flag is important for SELinux labeling.

#### 4. Access Portainer

Open browser:

```test
https://<raspberry-pi-ip>:9443
```

Accept self-signed certificate.

#### 5. Configure for Podman

1. Create admin account
2. Select "Get Started"
3. **Important:** Portainer may show some Docker-specific features that won't work with Podman

**Limitations:**

- Docker Swarm features won't work
- Some networking features may be limited
- Stacks work but with podman-compose backend

---

### Option 2: Cockpit with Podman (Recommended for Podman)

Cockpit provides native Podman support and is excellent for Raspberry Pi management.

#### 1. Install Cockpit

```bash
sudo apt install -y cockpit cockpit-podman
```

#### 2. Enable Cockpit

```bash
sudo systemctl enable --now cockpit.socket
```

#### 3. Access Cockpit

Open browser:

```text
https://<raspberry-pi-ip>:9090
```

Login with your Pi username and password.

#### 4. Navigate to Podman

Click "Podman containers" in the left sidebar.

**Benefits:**

- Native Podman support
- User and root containers
- Pod management
- Volume management
- Built-in monitoring
- Terminal access

---

### Option 3: CLI + Systemd (Most Control)

Manage everything via command line and systemd user services (covered in deployment section).

**Choose your preferred method and continue to the next section.**

---

## Deploying Langfuse

We'll cover three deployment methods:

1. Portainer Stacks (if using Portainer)
2. Podman Compose (Recommended)
3. Podman with Systemd (Most native)

---

### Method 1: Using Portainer Stacks

**Note:** If using Portainer with Podman, stack deployment works but uses podman-compose under the hood.

#### 1. Navigate to Stacks

In Portainer: `Stacks` → `Add stack`

#### 2. Name Your Stack

Name: `langfuse`

#### 3. Docker Compose Configuration

Use the same compose file as Method 2 below. Portainer will convert it for Podman.

---

### Method 2: Podman Compose (Recommended)

This is the most straightforward approach for Podman users.

#### 1. Create Project Directory

```bash
mkdir -p ~/langfuse
cd ~/langfuse
```

#### 2. Create docker-compose.yml

```bash
nano docker-compose.yml
```

#### 3. Paste Configuration

```yaml
version: '3.8'

services:
  langfuse-db:
    image: docker.io/postgres:15-alpine
    container_name: langfuse-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: changeme_secure_password
      POSTGRES_DB: langfuse
    volumes:
      - langfuse-db-data:/var/lib/postgresql/data:Z
    networks:
      - langfuse-network
    healthcheck:
      test: ['CMD-SHELL', 'pg_isready -U langfuse']
      interval: 10s
      timeout: 5s
      retries: 5

  langfuse:
    image: docker.io/langfuse/langfuse:latest
    container_name: langfuse-app
    restart: unless-stopped
    depends_on:
      langfuse-db:
        condition: service_healthy
    ports:
      - '3000:3000'
    environment:
      # Database Configuration
      DATABASE_URL: postgresql://langfuse:changeme_secure_password@langfuse-db:5432/langfuse

      # Required: Salt for encryption (generate a secure random string)
      SALT: your-random-salt-min-32-characters-long

      # Required: NextAuth secret (generate a secure random string)
      NEXTAUTH_SECRET: your-nextauth-secret-min-32-characters

      # Application URL (update with your Pi's IP)
      NEXTAUTH_URL: http://localhost:3000

      # Optional: Telemetry (set to false for privacy)
      TELEMETRY_ENABLED: false

      # Optional: Enable/disable features
      LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES: false

    networks:
      - langfuse-network
    healthcheck:
      test:
        [
          'CMD',
          'wget',
          '--no-verbose',
          '--tries=1',
          '--spider',
          'http://localhost:3000/api/public/health',
        ]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

volumes:
  langfuse-db-data:

networks:
  langfuse-network:
```

**Note:** Added `:Z` flag to PostgreSQL volume for SELinux compatibility.

#### 4. Generate Secure Secrets

```bash
# Generate SALT (32+ characters)
openssl rand -base64 32

# Generate NEXTAUTH_SECRET (32+ characters)
openssl rand -base64 32
```

#### 5. Update Configuration

Edit `docker-compose.yml` and replace:

- `changeme_secure_password` (in two places)
- `your-random-salt-min-32-characters-long`
- `your-nextauth-secret-min-32-characters`
- `localhost:3000` with your Pi's IP if accessing remotely

#### 6. Deploy with Podman Compose

```bash
podman-compose up -d
```

#### 7. Check Status

```bash
podman-compose ps
podman-compose logs -f
```

---

### Method 3: Podman with Systemd (Most Native)

This method uses systemd user services for automatic startup and management.

#### 1. Create Project Directory

```bash
mkdir -p ~/langfuse/{db,app}
cd ~/langfuse
```

#### 2. Create Pod

Pods group containers together (like Docker Compose services):

```bash
podman pod create \
  --name langfuse-pod \
  -p 3000:3000
```

#### 3. Start PostgreSQL Container

```bash
# Generate password
DB_PASSWORD=$(openssl rand -base64 32)
echo "Database password: $DB_PASSWORD" > ~/langfuse/credentials.txt
chmod 600 ~/langfuse/credentials.txt

# Create volume
podman volume create langfuse-db-data

# Run PostgreSQL
podman run -d \
  --pod langfuse-pod \
  --name langfuse-postgres \
  -e POSTGRES_USER=langfuse \
  -e POSTGRES_PASSWORD="$DB_PASSWORD" \
  -e POSTGRES_DB=langfuse \
  -v langfuse-db-data:/var/lib/postgresql/data:Z \
  --health-cmd "pg_isready -U langfuse" \
  --health-interval 10s \
  --health-retries 5 \
  docker.io/postgres:15-alpine
```

#### 4. Generate Application Secrets

```bash
SALT=$(openssl rand -base64 32)
NEXTAUTH_SECRET=$(openssl rand -base64 32)

echo "SALT: $SALT" >> ~/langfuse/credentials.txt
echo "NEXTAUTH_SECRET: $NEXTAUTH_SECRET" >> ~/langfuse/credentials.txt
```

#### 5. Start Langfuse Container

```bash
# Get your Pi's IP
PI_IP=$(hostname -I | awk '{print $1}')

podman run -d \
  --pod langfuse-pod \
  --name langfuse-app \
  -e DATABASE_URL="postgresql://langfuse:$DB_PASSWORD@127.0.0.1:5432/langfuse" \
  -e SALT="$SALT" \
  -e NEXTAUTH_SECRET="$NEXTAUTH_SECRET" \
  -e NEXTAUTH_URL="http://$PI_IP:3000" \
  -e TELEMETRY_ENABLED=false \
  docker.io/langfuse/langfuse:latest
```

**Note:** Containers in the same pod share localhost networking.

#### 6. Verify Running

```bash
podman pod ps
podman ps
```

#### 7. Generate Systemd Service Files

```bash
# Create systemd directory
mkdir -p ~/.config/systemd/user

# Generate service for the pod
podman generate systemd --new --files --name langfuse-pod

# Move to systemd directory
mv *.service ~/.config/systemd/user/

# Reload systemd
systemctl --user daemon-reload

# Enable services
systemctl --user enable pod-langfuse-pod.service

# Check status
systemctl --user status pod-langfuse-pod.service
```

Now the pod will start automatically on boot!

#### 8. Managing with Systemd

```bash
# Start
systemctl --user start pod-langfuse-pod.service

# Stop
systemctl --user stop pod-langfuse-pod.service

# Restart
systemctl --user restart pod-langfuse-pod.service

# View logs
journalctl --user -u pod-langfuse-pod.service -f
```

---

### Choosing a Method

- **Portainer:** If you want web UI and are comfortable with Docker Compose syntax
- **Podman Compose:** Best balance of ease and Podman features
- **Systemd:** Most native, best systemd integration, automatic startup

**Recommendation:** Start with Podman Compose (Method 2) for easiest setup.

---

## Configuration

### Environment Variables Reference

#### Database Settings

```yaml
DATABASE_URL: postgresql://user:password@host:port/database
```

**Components:**

- `user`: Database username
- `password`: Database password
- `host`: Database hostname (service name in Docker)
- `port`: PostgreSQL port (default: 5432)
- `database`: Database name

#### Security Settings

```yaml
SALT: <random-string-min-32-chars>
NEXTAUTH_SECRET: <random-string-min-32-chars>
```

**Requirements:**

- Both must be at least 32 characters
- Use cryptographically secure random values
- Never reuse between environments

#### Application URLs

```yaml
NEXTAUTH_URL: http://your-domain-or-ip:3000
```

**Usage:**

- Authentication callbacks
- OAuth redirect URIs
- Link generation

#### Optional Features

```yaml
# Disable telemetry
TELEMETRY_ENABLED: false

# Enable experimental features
LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES: true

# Set log level
LOG_LEVEL: info # debug, info, warn, error
```

---

## Accessing Langfuse

### 1. Initial Access

Open your browser:

```
http://<raspberry-pi-ip>:3000
```

**Example:** `http://192.168.1.100:3000`

### 2. First-Time Setup

1. You'll see the Langfuse welcome page
2. Click "Sign up" to create an account
3. Enter email and password
4. Complete registration

### 3. Create Your First Project

1. After login, create a new project
2. Note your API keys (shown once)
3. Configure your LLM application to use these keys

### 4. API Keys

Store your keys securely:

- **Public Key:** Used for ingesting data
- **Secret Key:** Full access, keep private

---

## Backup and Maintenance

### Database Backups

#### Manual Backup

```bash
# Create backup directory
mkdir -p ~/langfuse-backups

# Backup database
podman exec langfuse-postgres pg_dump -U langfuse langfuse > ~/langfuse-backups/langfuse-$(date +%Y%m%d-%H%M%S).sql
```

#### Automated Backup Script

Create backup script:

```bash
nano ~/langfuse-backup.sh
```

Add:

```bash
#!/bin/bash
BACKUP_DIR=~/langfuse-backups
DATE=$(date +%Y%m%d-%H%M%S)
mkdir -p $BACKUP_DIR

# Backup database
podman exec langfuse-postgres pg_dump -U langfuse langfuse | gzip > $BACKUP_DIR/langfuse-$DATE.sql.gz

# Keep only last 7 days
find $BACKUP_DIR -name "langfuse-*.sql.gz" -mtime +7 -delete

echo "Backup completed: langfuse-$DATE.sql.gz"
```

Make executable:

```bash
chmod +x ~/langfuse-backup.sh
```

#### Schedule Backups with Systemd Timer (Better than Cron for Podman)

Create timer unit:

```bash
mkdir -p ~/.config/systemd/user
nano ~/.config/systemd/user/langfuse-backup.service
```

Add:

```ini
[Unit]
Description=Langfuse Database Backup
Requires=pod-langfuse-pod.service
After=pod-langfuse-pod.service

[Service]
Type=oneshot
ExecStart=/home/pi/langfuse-backup.sh

[Install]
WantedBy=default.target
```

Create timer:

```bash
nano ~/.config/systemd/user/langfuse-backup.timer
```

Add:

```ini
[Unit]
Description=Daily Langfuse Backup Timer

[Timer]
OnCalendar=daily
OnCalendar=02:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable and start:

```bash
systemctl --user daemon-reload
systemctl --user enable langfuse-backup.timer
systemctl --user start langfuse-backup.timer

# Check timer status
systemctl --user list-timers
```

### Restore from Backup

```bash
# If using podman-compose
cd ~/langfuse
podman-compose down

# Restore database
gunzip -c ~/langfuse-backups/langfuse-YYYYMMDD-HHMMSS.sql.gz | podman exec -i langfuse-postgres psql -U langfuse langfuse

# If using podman-compose
podman-compose up -d

# If using systemd
systemctl --user start pod-langfuse-pod.service
```

### Update Langfuse

#### Via Podman Compose

```bash
cd ~/langfuse
podman-compose pull
podman-compose up -d
```

#### Via Systemd/Pod

```bash
# Stop services
systemctl --user stop pod-langfuse-pod.service

# Pull new images
podman pull docker.io/langfuse/langfuse:latest
podman pull docker.io/postgres:15-alpine

# Remove old containers (data persists in volumes)
podman rm -f langfuse-app langfuse-postgres

# Recreate with new images (repeat deployment commands)
# Then regenerate systemd files
cd ~/.config/systemd/user
rm pod-*.service container-*.service

podman generate systemd --new --files --name langfuse-pod
systemctl --user daemon-reload
systemctl --user enable pod-langfuse-pod.service
systemctl --user start pod-langfuse-pod.service
```

### Monitor Disk Usage

```bash
# Check Podman disk usage
podman system df

# Clean up unused data
podman system prune -a

# Clean up volumes (CAREFUL - will delete data!)
podman volume prune
```

### Log Management

```bash
# View logs (podman-compose)
podman-compose logs langfuse
podman-compose logs langfuse-db

# View logs (systemd)
journalctl --user -u pod-langfuse-pod.service -f
journalctl --user -u container-langfuse-app.service -f

# View logs (direct podman)
podman logs langfuse-app -f
podman logs langfuse-postgres -f
```

Configure log size limits in compose file:

```yaml
services:
  langfuse:
    logging:
      driver: 'journald'
      options:
        tag: 'langfuse-app'
```

Or for individual containers:

```bash
podman run -d \
  --log-driver journald \
  --log-opt tag=langfuse-app \
  ...
```

---

## Troubleshooting

### Common Issues

#### 1. Langfuse Won't Start

**Symptom:** Container keeps restarting

**Check logs:**

```bash
# Podman compose
podman-compose logs langfuse

# Direct podman
podman logs langfuse-app

# Systemd
journalctl --user -u container-langfuse-app.service -f
```

**Common causes:**

- Database not ready (wait 60 seconds)
- Invalid environment variables
- Memory issues on Raspberry Pi
- Rootless permission issues

**Solution:**

```bash
# Check container status
podman ps -a

# Verify database is healthy
podman exec langfuse-postgres pg_isready -U langfuse

# Restart (podman-compose)
podman-compose restart

# Restart (systemd)
systemctl --user restart pod-langfuse-pod.service
```

#### 2. Cannot Connect to Database

**Error:** `connection refused` or `could not connect to server`

**Solution:**

```bash
# Check if PostgreSQL is running
podman ps | grep postgres

# For pod deployment, containers share localhost
# Verify DATABASE_URL uses localhost or service name

# Check network (podman-compose)
podman network ls
podman network inspect langfuse_langfuse-network

# For pod deployment, verify pod is running
podman pod ps
```

#### 3. Permission Denied Errors (Rootless Specific)

**Error:** Permission errors with volumes or files

**Solution:**

```bash
# Ensure SELinux labels on volumes (use :Z flag)
# In compose file:
volumes:
  - langfuse-db-data:/var/lib/postgresql/data:Z

# Check subuid/subgid mappings
grep $USER /etc/subuid
grep $USER /etc/subgid

# Reset podman if needed
podman system reset  # WARNING: Deletes all containers and volumes!
```

#### 4. Out of Memory

**Symptom:** Containers randomly stopping, system freezing

**Check memory:**

```bash
free -h
podman stats
```

**Solutions:**

```bash
# Add swap file (2GB example)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Limit container memory in compose file
```

Add to langfuse service:

```yaml
mem_limit: 512m
mem_reservation: 256m
```

Or for direct podman:

```bash
podman run -d \
  --memory=512m \
  --memory-reservation=256m \
  ...
```

#### 5. Port Already in Use

**Error:** `port is already allocated`

**Solution:**

```bash
# Find what's using port 3000
sudo lsof -i :3000

# Or use ss
ss -tulpn | grep :3000

# Kill the process or change Langfuse port
```

Change port in compose file:

```yaml
ports:
  - '3001:3000' # Use 3001 instead
```

#### 6. Podman Socket Issues

**Error:** Cannot connect to Podman socket

**Solution:**

```bash
# Check socket status
systemctl --user status podman.socket

# Restart socket
systemctl --user restart podman.socket

# Verify socket path
ls -la /run/user/$(id -u)/podman/podman.sock

# Enable lingering
loginctl enable-linger $USER
```

#### 7. Image Pull Failures

**Error:** Cannot pull images

**Solution:**

```bash
# Check registry configuration
cat ~/.config/containers/registries.conf

# Try pulling manually with full path
podman pull docker.io/langfuse/langfuse:latest

# Check network connectivity
ping docker.io

# Clear cache and retry
podman system prune -a
```

#### 8. Systemd Service Won't Start

**Error:** Systemd service fails to start

**Solution:**

```bash
# Check service status
systemctl --user status pod-langfuse-pod.service

# View detailed logs
journalctl --user -u pod-langfuse-pod.service -n 50

# Regenerate systemd files
cd ~/.config/systemd/user
rm pod-*.service container-*.service
podman generate systemd --new --files --name langfuse-pod
systemctl --user daemon-reload
systemctl --user restart pod-langfuse-pod.service

# Verify lingering is enabled
loginctl show-user $USER | grep Linger
```

#### 9. Can't Access from Other Devices

**Check firewall:**

```bash
sudo ufw status
sudo ufw allow 3000/tcp
```

**Check Podman port binding:**

For podman-compose, ensure:

```yaml
ports:
  - '0.0.0.0:3000:3000' # Explicit binding
```

For direct podman:

```bash
podman run -d \
  -p 0.0.0.0:3000:3000 \
  ...
```

#### 10. Volume Data Loss

**Symptom:** Data disappears after restart

**Verify volumes:**

```bash
podman volume ls | grep langfuse
podman volume inspect langfuse-db-data

# Check mount point
podman volume inspect langfuse-db-data | grep Mountpoint
```

**Ensure proper volume declaration in compose file:**

```yaml
volumes:
  langfuse-db-data: # Named volume
```

#### 11. Rootless vs Root Confusion

**Issue:** Mixing root and rootless containers

**Solution:**

```bash
# List rootless containers
podman ps -a

# List root containers (need sudo)
sudo podman ps -a

# Stick to one approach - rootless is recommended
# Never mix root and rootless for the same application
```

---

## Performance Optimization

### 1. Use SSD Instead of SD Card

**Best improvement for Raspberry Pi:**

```bash
# Boot from USB SSD
# Follow official Raspberry Pi documentation for boot order
```

### 2. PostgreSQL Tuning

Add to postgres service environment:

```yaml
environment:
  # Reduce memory for Raspberry Pi
  POSTGRES_SHARED_BUFFERS: 128MB
  POSTGRES_EFFECTIVE_CACHE_SIZE: 256MB
  POSTGRES_WORK_MEM: 4MB
  POSTGRES_MAINTENANCE_WORK_MEM: 64MB
```

### 3. Podman-Specific Optimizations

#### Use Host Network for Better Performance (Careful)

```yaml
services:
  langfuse:
    network_mode: host
    environment:
      # Must update ports since we're on host network
      PORT: 3000
```

**Note:** This bypasses container networking overhead but exposes all ports.

#### Enable CDI (Container Device Interface)

For better hardware acceleration:

```bash
nano ~/.config/containers/containers.conf
```

Add:

```toml
[engine]
cgroup_manager = "systemd"
events_logger = "journald"

[containers]
# Enable cgroup v2 features
netns = "host"  # Only if using host networking
```

### 4. Storage Driver Optimization

Check current driver:

```bash
podman info | grep graphDriverName
```

Overlay2 is usually best. If using VFS, switch:

```bash
nano ~/.config/containers/storage.conf
```

Add:

```toml
[storage]
driver = "overlay"

[storage.options.overlay]
mount_program = "/usr/bin/fuse-overlayfs"
```

### 5. Enable Podman Restart Service

Ensure containers restart after reboot:

```bash
# Enable socket activation
systemctl --user enable podman.socket
systemctl --user enable podman-restart.service

# Enable auto-update for systemd services
systemctl --user enable podman-auto-update.timer
```

### 6. Resource Limits

In compose file:

```yaml
services:
  langfuse:
    deploy:
      resources:
        limits:
          cpus: '1.5'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
```

### 7. Use Podman Auto-Update

Tag containers for auto-update:

```bash
podman run -d \
  --label "io.containers.autoupdate=registry" \
  docker.io/langfuse/langfuse:latest
```

Enable auto-update timer:

```bash
systemctl --user enable podman-auto-update.timer
systemctl --user start podman-auto-update.timer

# Check status
systemctl --user status podman-auto-update.timer
```

### 8. Monitoring Resources

```bash
# Real-time stats
podman stats

# System resource usage
podman system df

# Container resource limits
podman inspect langfuse-app | grep -A 20 HostConfig
```

### 9. Optimize Journald Logging

```bash
sudo nano /etc/systemd/journald.conf
```

Add:

```ini
[Journal]
SystemMaxUse=500M
SystemKeepFree=1G
MaxFileSec=1week
```

Restart journald:

```bash
sudo systemctl restart systemd-journald
```

---

## Advanced Configuration

### External Access with Cloudflare Tunnel

For secure external access without opening ports:

Add to your compose file:

```yaml
cloudflared:
  image: docker.io/cloudflare/cloudflared:latest
  command: tunnel --no-autoupdate run
  environment:
    TUNNEL_TOKEN: your-tunnel-token
  networks:
    - langfuse-network
```

Or with Podman directly:

```bash
podman run -d \
  --pod langfuse-pod \
  --name cloudflared \
  -e TUNNEL_TOKEN=your-tunnel-token \
  docker.io/cloudflare/cloudflared:latest \
  tunnel --no-autoupdate run
```

### Email Configuration (SMTP)

For notifications and user invites:

```yaml
# Add to langfuse service environment
SMTP_CONNECTION_URL: smtp://username:password@smtp.gmail.com:587
EMAIL_FROM_NAME: Langfuse
EMAIL_FROM_ADDRESS: noreply@yourdomain.com
```

### Custom Domain Setup

1. Point domain A record to your Raspberry Pi
2. Set up reverse proxy (Nginx/Caddy)
3. Configure SSL with Let's Encrypt
4. Update NEXTAUTH_URL

#### Nginx Reverse Proxy

Install Nginx:

```bash
sudo apt install nginx certbot python3-certbot-nginx
```

Create config:

```bash
sudo nano /etc/nginx/sites-available/langfuse
```

Add:

```nginx
server {
    listen 80;
    server_name langfuse.yourdomain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Enable:

```bash
sudo ln -s /etc/nginx/sites-available/langfuse /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

Get SSL certificate:

```bash
sudo certbot --nginx -d langfuse.yourdomain.com
```

### Running as Root (Not Recommended)

If you absolutely need root Podman:

```bash
# All commands need sudo
sudo podman ps
sudo podman-compose up -d

# Systemd services go in /etc/systemd/system
sudo systemctl enable pod-langfuse-pod
```

**Note:** Rootless is more secure and recommended.

### Multiple Instance Isolation

Run multiple Langfuse instances:

**Instance 1 (Production):**

```bash
mkdir -p ~/langfuse-prod
cd ~/langfuse-prod
# Use port 3000
podman-compose up -d
```

**Instance 2 (Development):**

```bash
mkdir -p ~/langfuse-dev
cd ~/langfuse-dev
# Edit compose file, use port 3001
podman-compose up -d
```

### Database Connection Pooling with PgBouncer

Add to compose file:

```yaml
pgbouncer:
  image: docker.io/edoburu/pgbouncer:latest
  environment:
    DATABASE_URL: postgresql://langfuse:password@langfuse-db:5432/langfuse
    POOL_MODE: transaction
    MAX_CLIENT_CONN: 100
    DEFAULT_POOL_SIZE: 20
  networks:
    - langfuse-network
```

Update Langfuse DATABASE_URL:

```yaml
DATABASE_URL: postgresql://langfuse:password@pgbouncer:5432/langfuse
```

### Using Podman Secrets

For better security, use Podman secrets:

```bash
# Create secrets
echo -n "my-db-password" | podman secret create db_password -
echo -n "my-salt" | podman secret create langfuse_salt -
echo -n "my-nextauth-secret" | podman secret create nextauth_secret -

# Reference in compose file
```

```yaml
services:
  langfuse-db:
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password

  langfuse:
    environment:
      SALT_FILE: /run/secrets/langfuse_salt
      NEXTAUTH_SECRET_FILE: /run/secrets/nextauth_secret
    secrets:
      - langfuse_salt
      - nextauth_secret

secrets:
  db_password:
    external: true
  langfuse_salt:
    external: true
  nextauth_secret:
    external: true
```

### Monitoring with Prometheus

Create monitoring stack:

```yaml
prometheus:
  image: docker.io/prom/prometheus:latest
  ports:
    - '9090:9090'
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml:Z
    - prometheus-data:/prometheus:Z
  networks:
    - langfuse-network

grafana:
  image: docker.io/grafana/grafana:latest
  ports:
    - '3001:3000'
  volumes:
    - grafana-data:/var/lib/grafana:Z
  networks:
    - langfuse-network
```

---

## Security Best Practices

### 1. Use Rootless Podman (Default)

**Already doing this!** Podman's rootless mode is inherently more secure:

```bash
# Verify you're running rootless
podman info | grep rootless
# Should show: rootless: true

# Your containers run as your user, not root
ps aux | grep podman
```

**Benefits:**

- Containers can't escape to root access
- No privileged daemon
- Better multi-user isolation

### 2. Change Default Passwords

Never use example passwords in production:

```bash
# Generate strong passwords
openssl rand -base64 32
```

Update in compose file:

```yaml
POSTGRES_PASSWORD: $(openssl rand -base64 32)
```

### 3. Use Podman Secrets

Instead of environment variables:

```bash
echo -n "my-secret-password" | podman secret create db_password -
```

Reference in compose:

```yaml
services:
  langfuse-db:
    secrets:
      - db_password
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password

secrets:
  db_password:
    external: true
```

### 4. Network Isolation

Keep database internal (no exposed ports):

```yaml
langfuse-db:
  # DO NOT expose ports externally
  # No "ports:" section for database
  # Only accessible within the pod/network
```

For pod deployment, database is automatically isolated to localhost.

### 5. SELinux Labels

Always use `:Z` for volume mounts in rootless mode:

```yaml
volumes:
  - langfuse-db-data:/var/lib/postgresql/data:Z
```

This ensures proper SELinux labeling and prevents permission issues.

### 6. Regular Updates

```bash
# Update system
sudo apt update && sudo apt upgrade

# Update containers (podman-compose)
cd ~/langfuse
podman-compose pull
podman-compose up -d

# Or use auto-update
systemctl --user enable podman-auto-update.timer
```

### 7. Firewall Configuration

```bash
# Enable firewall
sudo ufw enable

# Allow only necessary ports
sudo ufw allow 22/tcp   # SSH
sudo ufw allow from 192.168.1.0/24 to any port 3000  # Langfuse (local network only)
sudo ufw allow 9443/tcp # Portainer (if used)
sudo ufw allow 9090/tcp # Cockpit (if used)

# Check status
sudo ufw status numbered
```

### 8. Limit External Access

Don't expose to internet unless necessary:

```yaml
# Bind only to localhost
ports:
  - "127.0.0.1:3000:3000"

# Or local network
ports:
  - "192.168.1.100:3000:3000"
```

### 9. Backup Encryption

```bash
# Encrypt backups
gpg --symmetric ~/langfuse-backups/backup.sql

# Decrypt when needed
gpg --decrypt backup.sql.gpg > backup.sql
```

### 10. Resource Limits

Prevent resource exhaustion attacks:

```yaml
services:
  langfuse:
    deploy:
      resources:
        limits:
          cpus: '1.5'
          memory: 512M
```

### 11. Disable Unnecessary Features

```yaml
environment:
  TELEMETRY_ENABLED: false
  LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES: false
```

### 12. Monitor Logs

Set up log monitoring:

```bash
# Watch for failed auth attempts
journalctl --user -u pod-langfuse-pod -f | grep -i "failed\|error\|unauthorized"

# Set up log rotation
sudo nano /etc/systemd/journald.conf
```

```ini
[Journal]
SystemMaxUse=500M
MaxRetentionSec=1week
```

### 13. User Namespace Remapping

Already done in rootless Podman! Verify:

```bash
podman unshare cat /proc/self/uid_map
```

This shows how UIDs are remapped from container to host.

### 14. Scan Images for Vulnerabilities

```bash
# Install scanner
sudo apt install trivy

# Scan image
trivy image docker.io/langfuse/langfuse:latest
trivy image docker.io/postgres:15-alpine
```

### 15. Disable Privilege Escalation

In compose file:

```yaml
services:
  langfuse:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
```

### 16. Read-Only Root Filesystem

For extra hardening:

```yaml
services:
  langfuse:
    read_only: true
    tmpfs:
      - /tmp
      - /run
```

**Note:** May require additional configuration for Langfuse.

### 17. Audit Podman Events

Monitor container activity:

```bash
# Start event monitoring
podman events --since 1h

# Log events
podman events > ~/podman-events.log &
```

### Security Checklist

- [ ] Running rootless Podman
- [ ] All default passwords changed
- [ ] Secrets properly configured
- [ ] Database not exposed externally
- [ ] Firewall configured
- [ ] Regular updates scheduled
- [ ] Backups encrypted
- [ ] Resource limits set
- [ ] Logs monitored
- [ ] Images scanned for vulnerabilities

---

## Useful Commands Reference

### Container Management

```bash
# View all containers
podman ps -a

# View only running containers
podman ps

# Start a container
podman start langfuse-app

# Stop a container
podman stop langfuse-app

# Restart a container
podman restart langfuse-app

# Remove a container
podman rm langfuse-app

# Force remove a running container
podman rm -f langfuse-app
```

### Logs

```bash
# View logs (podman-compose)
podman-compose logs langfuse
podman-compose logs langfuse-db

# View logs (direct)
podman logs langfuse-app

# Follow logs in real-time
podman logs -f langfuse-app

# View last 50 lines
podman logs --tail 50 langfuse-app

# View logs since timestamp
podman logs --since 2024-01-01T10:00:00 langfuse-app

# Systemd logs
journalctl --user -u pod-langfuse-pod.service -f
journalctl --user -u container-langfuse-app.service --since today
```

### Exec into Containers

```bash
# Shell into Langfuse
podman exec -it langfuse-app sh

# Shell into PostgreSQL
podman exec -it langfuse-postgres sh

# Run psql directly
podman exec -it langfuse-postgres psql -U langfuse

# Run a single command
podman exec langfuse-postgres pg_isready -U langfuse
```

### Compose Commands

```bash
# Start all services
podman-compose up -d

# Stop all services
podman-compose down

# Restart all services
podman-compose restart

# View status
podman-compose ps

# Pull latest images
podman-compose pull

# Rebuild and restart
podman-compose up -d --build

# Remove everything including volumes
podman-compose down -v
```

### Pod Management

```bash
# List pods
podman pod ps

# Create pod
podman pod create --name myp od -p 8080:80

# Start pod
podman pod start langfuse-pod

# Stop pod
podman pod stop langfuse-pod

# Remove pod (removes containers too)
podman pod rm langfuse-pod

# Inspect pod
podman pod inspect langfuse-pod

# View pod logs
podman pod logs langfuse-pod
```

### Volume Management

```bash
# List volumes
podman volume ls

# Inspect volume
podman volume inspect langfuse-db-data

# Create volume
podman volume create myvolume

# Remove volume
podman volume rm langfuse-db-data

# Prune unused volumes (CAREFUL!)
podman volume prune

# View volume mount point
podman volume inspect langfuse-db-data | grep Mountpoint
```

### Network Management

```bash
# List networks
podman network ls

# Inspect network
podman network inspect langfuse_langfuse-network

# Create network
podman network create mynetwork

# Remove network
podman network rm mynetwork

# Connect container to network
podman network connect mynetwork container-name

# Disconnect container from network
podman network disconnect mynetwork container-name
```

### Image Management

```bash
# List images
podman images

# Pull image
podman pull docker.io/langfuse/langfuse:latest

# Remove image
podman rmi langfuse/langfuse:latest

# Remove unused images
podman image prune

# Remove all images
podman image prune -a

# Tag image
podman tag langfuse/langfuse:latest mylangfuse:v1
```

### System Management

```bash
# System information
podman info

# System disk usage
podman system df

# Clean up everything
podman system prune -a

# Reset Podman (WARNING: Removes everything!)
podman system reset

# Check Podman events
podman events

# Check resource usage
podman stats

# Check specific container resources
podman stats langfuse-app
```

### Systemd Commands

```bash
# Enable service (start on boot)
systemctl --user enable pod-langfuse-pod.service

# Disable service
systemctl --user disable pod-langfuse-pod.service

# Start service
systemctl --user start pod-langfuse-pod.service

# Stop service
systemctl --user stop pod-langfuse-pod.service

# Restart service
systemctl --user restart pod-langfuse-pod.service

# Check status
systemctl --user status pod-langfuse-pod.service

# View logs
journalctl --user -u pod-langfuse-pod.service

# Follow logs
journalctl --user -u pod-langfuse-pod.service -f

# Reload systemd daemon
systemctl --user daemon-reload

# List all user services
systemctl --user list-units --type=service

# Check if lingering is enabled
loginctl show-user $USER | grep Linger
```

### Generate Systemd Units

```bash
# Generate for a container
podman generate systemd --new --files --name langfuse-app

# Generate for a pod
podman generate systemd --new --files --name langfuse-pod

# Generate with restart policy
podman generate systemd --new --restart-policy=always --files --name langfuse-app

# Generate and place in systemd directory
cd ~/.config/systemd/user
podman generate systemd --new --files --name langfuse-pod
systemctl --user daemon-reload
```

### Inspection and Debugging

```bash
# Inspect container
podman inspect langfuse-app

# Get container IP
podman inspect langfuse-app | grep IPAddress

# Check container processes
podman top langfuse-app

# Check container ports
podman port langfuse-app

# View container changes
podman diff langfuse-app

# Health check
podman healthcheck run langfuse-app

# Copy files from container
podman cp langfuse-app:/path/in/container /path/on/host

# Copy files to container
podman cp /path/on/host langfuse-app:/path/in/container
```

### Auto-Update

```bash
# Check auto-update status
systemctl --user status podman-auto-update.timer

# Run auto-update manually
podman auto-update

# Dry run (see what would update)
podman auto-update --dry-run

# Enable auto-update timer
systemctl --user enable --now podman-auto-update.timer
```

### Socket Management

```bash
# Check socket status
systemctl --user status podman.socket

# Start socket
systemctl --user start podman.socket

# Enable socket
systemctl --user enable podman.socket

# View socket path
echo $XDG_RUNTIME_DIR/podman/podman.sock

# Test socket
curl -s --unix-socket /run/user/$(id -u)/podman/podman.sock http://localhost/v1.0.0/libpod/info
```

### Quick Troubleshooting Commands

```bash
# Check what's wrong with a container
podman ps -a
podman logs containername
podman inspect containername

# Network troubleshooting
podman network ls
podman network inspect networkname

# Volume troubleshooting
podman volume ls
podman volume inspect volumename

# Permission troubleshooting
podman unshare ls -la /path/to/volume

# Reset and start fresh (NUCLEAR OPTION)
podman system reset
```

---

## Conclusion

You now have a fully functional self-hosted Langfuse instance running on your Raspberry Pi with Podman, offering a secure, rootless container deployment. This setup provides:

- ✅ Full control over your LLM observability data
- ✅ Rootless, daemonless container architecture (more secure than Docker)
- ✅ Easy management via Portainer, Cockpit, or CLI
- ✅ Persistent data storage with proper volume management
- ✅ Systemd integration for automatic startup
- ✅ Automated backups capability
- ✅ Scalable architecture

### Next Steps

1. Configure your LLM applications to use Langfuse API keys
2. Set up automated backups with systemd timers
3. Monitor resource usage with `podman stats`
4. Consider setting up external access if needed
5. Explore Cockpit for comprehensive system management

### Advantages of Podman Over Docker

- **Security:** Rootless by default, no daemon running as root
- **Systemd Integration:** Native support for systemd units
- **Resource Efficiency:** Lower overhead without daemon
- **Compatibility:** Drop-in replacement for Docker commands
- **Pod Support:** Native pod concept (like Kubernetes)

### Resources

- [Langfuse Documentation](https://langfuse.com/docs)
- [Podman Documentation](https://docs.podman.io)
- [Podman Compose](https://github.com/containers/podman-compose)
- [Portainer Documentation](https://docs.portainer.io)
- [Cockpit Documentation](https://cockpit-project.org/documentation.html)
- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation)

### Getting Help

- Langfuse GitHub: https://github.com/langfuse/langfuse
- Langfuse Discord: https://discord.gg/7NXusRtqYU
- Podman Discussions: https://github.com/containers/podman/discussions
- Raspberry Pi Forums: https://forums.raspberrypi.com

### Podman-Specific Resources

- Podman Tutorials: https://docs.podman.io/en/latest/Tutorials.html
- Rootless Containers: https://github.com/containers/podman/blob/main/docs/tutorials/rootless_tutorial.md
- Systemd Integration: https://docs.podman.io/en/latest/markdown/podman-generate-systemd.1.html

---

**Document Version:** 2.0 (Podman Edition)  
**Last Updated:** January 2026  
**Tested On:** Raspberry Pi 4 (4GB), Raspberry Pi OS 64-bit, Podman 4.x+

### Quick Reference Card

```bash
# Start Langfuse
podman-compose up -d                    # or
systemctl --user start pod-langfuse-pod

# Stop Langfuse
podman-compose down                     # or
systemctl --user stop pod-langfuse-pod

# View logs
podman-compose logs -f                  # or
journalctl --user -u pod-langfuse-pod -f

# Backup database
podman exec langfuse-postgres pg_dump -U langfuse langfuse > backup.sql

# Update Langfuse
podman-compose pull && podman-compose up -d

# Check resource usage
podman stats

# Troubleshoot
podman ps -a
podman logs langfuse-app
systemctl --user status pod-langfuse-pod
```
