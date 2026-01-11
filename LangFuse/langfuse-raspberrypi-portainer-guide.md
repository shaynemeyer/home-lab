# Complete Guide: Self-Hosted Langfuse on Raspberry Pi with Portainer (Podman Backend)

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Hardware Requirements](#hardware-requirements)
4. [Initial Raspberry Pi Setup](#initial-raspberry-pi-setup)
5. [Installing Podman](#installing-podman)
6. [Installing Portainer](#installing-portainer)
7. [Deploying Langfuse via Portainer](#deploying-langfuse-via-portainer)
8. [Managing Langfuse in Portainer](#managing-langfuse-in-portainer)
9. [Configuration](#configuration)
10. [Backup and Maintenance](#backup-and-maintenance)
11. [Monitoring and Logs](#monitoring-and-logs)
12. [Troubleshooting](#troubleshooting)
13. [Advanced Portainer Features](#advanced-portainer-features)

---

## Overview

This guide provides a complete walkthrough for deploying Langfuse (an open-source LLM engineering platform) on a Raspberry Pi using Portainer as your primary management interface, with Podman as the container runtime.

**What you'll achieve:**

- Self-hosted Langfuse instance on Raspberry Pi
- Web-based management via Portainer GUI
- Rootless, secure container deployment with Podman
- Point-and-click container management
- Easy updates and rollbacks
- Persistent data storage

**Estimated time:** 45-60 minutes

**Why This Stack?**

- **Portainer:** Beautiful web UI for container management
- **Podman:** Rootless, daemonless, secure Docker alternative
- **Raspberry Pi:** Low-cost, low-power self-hosting platform

---

## Prerequisites

### Software Requirements

- Raspberry Pi OS (64-bit recommended)
- SSH access or keyboard/monitor access
- Internet connection
- Modern web browser

### Knowledge Requirements

- Basic Linux command line navigation
- Understanding of IP addresses
- Familiarity with web interfaces

### Before You Start

- Ensure your Raspberry Pi is updated
- Have at least 8GB free storage
- Know your Raspberry Pi's IP address (run `hostname -I` to find it)

---

## Hardware Requirements

### Minimum Requirements

- **Model:** Raspberry Pi 4 (2GB RAM)
- **Storage:** 16GB microSD card
- **Network:** Ethernet or WiFi connection

### Recommended Setup

- **Model:** Raspberry Pi 4 or 5 (4GB+ RAM)
- **Storage:** 32GB+ microSD card or USB SSD
- **Network:** Ethernet for stability
- **Cooling:** Heat sink or active cooling

### Why These Specs?

- Langfuse + PostgreSQL need sufficient RAM
- Portainer adds minimal overhead
- Multiple containers running simultaneously

---

## Initial Raspberry Pi Setup

### 1. Update Your System

```bash
sudo apt update && sudo apt upgrade -y
```

**Wait for this to complete - it may take 5-10 minutes.**

### 2. Configure Memory Split (Headless Setup)

For servers without a desktop, reduce GPU memory:

```bash
sudo raspi-config
```

Navigate: `Performance Options` → `GPU Memory` → Set to `16` → `Finish`

### 3. Enable Container Features

Edit boot configuration:

```bash
sudo nano /boot/firmware/cmdline.txt
```

Add to the **end** of the existing line (don't create a new line):

```bash
cgroup_enable=cpuset cgroup_enable=memory cgroup_memory=1
```

Press `Ctrl+X`, then `Y`, then `Enter` to save.

**Reboot:**

```bash
sudo reboot
```

Wait 1-2 minutes, then reconnect via SSH.

### 4. Set Static IP (Recommended)

This ensures your Pi always has the same IP address.

Edit network configuration:

```bash
sudo nano /etc/dhcpcd.conf
```

Scroll to the bottom and add (adjust values for your network):

```bash
interface eth0
static ip_address=192.168.1.100/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1 8.8.8.8
```

**For WiFi, use `interface wlan0` instead of `eth0`.**

Save and restart networking:

```bash
sudo service dhcpcd restart
```

Test your new IP:

```bash
hostname -I
```

---

## Installing Podman

Podman is a Docker alternative that runs containers without a daemon and supports rootless operation (more secure).

### 1. Install Podman

```bash
sudo apt update
sudo apt install -y podman podman-compose
```

### 2. Verify Installation

```bash
podman --version
```

You should see something like `podman version 4.x.x`.

### 3. Configure Rootless Podman

Enable user namespaces:

```bash
# Check if already enabled
cat /proc/sys/kernel/unprivileged_userns_clone
```

If it returns `0`, enable it:

```bash
echo 'kernel.unprivileged_userns_clone=1' | sudo tee /etc/sysctl.d/00-local-userns.conf
sudo sysctl -p /etc/sysctl.d/00-local-userns.conf
```

### 4. Configure User ID Mappings

```bash
# Check current mappings
grep $USER /etc/subuid
grep $USER /etc/subgid
```

If empty or missing, add them:

```bash
sudo usermod --add-subuids 100000-165535 --add-subgids 100000-165535 $USER
```

**Log out and back in** for changes to take effect.

### 5. Enable Podman Socket for Portainer

The Podman socket allows Portainer to communicate with Podman:

```bash
# Enable and start the socket
systemctl --user enable podman.socket
systemctl --user start podman.socket
```

Verify it's running:

```bash
systemctl --user status podman.socket
```

You should see **"active (listening)"** in green.

### 6. Enable Lingering

This ensures your containers start at boot even if you're not logged in:

```bash
loginctl enable-linger $USER
```

### 7. Test Podman

```bash
podman run --rm hello-world
```

You should see a "Hello from Docker!" message (Podman is Docker-compatible).

### 8. Configure Container Registries

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

Save with `Ctrl+X`, `Y`, `Enter`.

---

## Installing Portainer

Portainer provides a beautiful web interface for managing containers.

### 1. Create Portainer Data Volume

```bash
podman volume create portainer_data
```

### 2. Deploy Portainer Container

```bash
podman run -d \
  -p 9443:9443 \
  --name portainer \
  --restart=always \
  -v /run/user/$(id -u)/podman/podman.sock:/var/run/docker.sock:Z \
  -v portainer_data:/data:Z \
  docker.io/portainer/portainer-ce:latest
```

**What this does:**

- `-d`: Run in background
- `-p 9443:9443`: Expose Portainer web interface on port 9443
- `--name portainer`: Name the container
- `--restart=always`: Auto-restart if it crashes
- `-v ... /podman.sock`: Connect to Podman socket
- `-v portainer_data:/data`: Persistent data storage
- `:Z`: SELinux label (important for security)

### 3. Wait for Portainer to Start

```bash
podman logs -f portainer
```

Wait until you see messages about "Starting Portainer" (about 30 seconds).

Press `Ctrl+C` to stop viewing logs.

### 4. Access Portainer Web Interface

Open your web browser and navigate to:

```text
https://<your-raspberry-pi-ip>:9443
```

**Example:** `https://192.168.1.100:9443`

**You'll see a security warning** - this is normal because Portainer uses a self-signed certificate. Click "Advanced" → "Proceed" (or similar depending on your browser).

### 5. Initial Portainer Setup

You'll see the Portainer setup screen:

1. **Create admin account:**
   - Username: `admin` (or your choice)
   - Password: Choose a strong password (min 12 characters)
2. Click **"Create user"**

3. On the next screen, click **"Get Started"**

4. You'll see the "Environments" page - click on **"local"**

### 6. Verify Podman Connection

You should now see the Portainer dashboard showing:

- Container count
- Image count
- Volume count
- Network count

**If you see this, Portainer is successfully connected to Podman!**

---

## Deploying Langfuse via Portainer

Now we'll deploy Langfuse using Portainer's Stack feature (equivalent to Docker Compose).

### Step 1: Navigate to Stacks

In Portainer's left sidebar:

1. Click **"Stacks"**
2. Click **"+ Add stack"** button

### Step 2: Name Your Stack

In the "Name" field, enter:

```bash
langfuse
```

### Step 3: Prepare Your Secrets

Before pasting the configuration, we need to generate secure secrets.

**Open a new SSH session** (or terminal on Pi) and run:

```bash
# Generate SALT (copy the output)
openssl rand -base64 32

# Generate NEXTAUTH_SECRET (copy the output)
openssl rand -base64 32

# Generate database password (copy the output)
openssl rand -base64 32
```

**Keep these values handy - you'll need them in the next step.**

### Step 4: Paste Docker Compose Configuration

Back in Portainer, select **"Web editor"** and paste:

```yaml
version: '3.8'

services:
  langfuse-db:
    image: docker.io/postgres:15-alpine
    container_name: langfuse-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: REPLACE_WITH_DB_PASSWORD
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
      DATABASE_URL: postgresql://langfuse:REPLACE_WITH_DB_PASSWORD@langfuse-db:5432/langfuse

      # Required: Salt for encryption
      SALT: REPLACE_WITH_SALT

      # Required: NextAuth secret
      NEXTAUTH_SECRET: REPLACE_WITH_NEXTAUTH_SECRET

      # Application URL - UPDATE THIS WITH YOUR PI'S IP
      NEXTAUTH_URL: http://192.168.1.100:3000

      # Optional: Telemetry
      TELEMETRY_ENABLED: false

      # Optional: Features
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

### Step 5: Replace Placeholder Values

Now update the configuration with your generated secrets:

1. **Find and replace** `REPLACE_WITH_DB_PASSWORD` (appears **twice**) with your database password
2. **Find and replace** `REPLACE_WITH_SALT` with your generated SALT
3. **Find and replace** `REPLACE_WITH_NEXTAUTH_SECRET` with your generated NEXTAUTH_SECRET
4. **Update** `NEXTAUTH_URL` with your Raspberry Pi's actual IP address

**Example after replacement:**

```yaml
POSTGRES_PASSWORD: x8Jk3mP9qR2vN7yH4tL6sW1eB5oI0uA8
DATABASE_URL: postgresql://langfuse:x8Jk3mP9qR2vN7yH4tL6sW1eB5oI0uA8@langfuse-db:5432/langfuse
SALT: Z9xC8vB7nM6qW5eR4tY3uI2oP1aS0dF6
NEXTAUTH_SECRET: A1sD2fG3hJ4kL5zX6cV7bN8mQ9wE0rT5
NEXTAUTH_URL: http://192.168.1.100:3000
```

### Step 6: Deploy the Stack

Scroll down and click **"Deploy the stack"** button.

### Step 7: Monitor Deployment

Portainer will now:

1. Pull the PostgreSQL image (~30 seconds)
2. Pull the Langfuse image (~1-2 minutes)
3. Create volumes
4. Create network
5. Start containers

You'll be redirected to the stack details page. Watch the status:

- **Status should show:** "Active" with a green indicator
- **Containers should show:** 2 running containers

### Step 8: Check Container Health

Click on **"Containers"** in the left sidebar.

You should see:

- **langfuse-postgres** - Status: "healthy" (green)
- **langfuse-app** - Status: "running" or "healthy" (green)

**If you see yellow or red status**, wait 1-2 minutes for initialization, then refresh.

### Step 9: View Logs

To check if Langfuse started successfully:

1. Click on **"langfuse-app"** container
2. Click **"Logs"** tab
3. Look for messages like:
   - "Server listening on port 3000"
   - "Database connected"

**If you see errors**, proceed to the Troubleshooting section.

---

## Accessing Langfuse

### 1. Open Langfuse Web Interface

Open your browser and navigate to:

```text
http://<your-raspberry-pi-ip>:3000
```

**Example:** `http://192.168.1.100:3000`

### 2. First-Time Setup

You should see the Langfuse welcome page:

1. Click **"Sign up"**
2. Enter your email address
3. Create a password
4. Click **"Sign up"**

### 3. Create Your First Project

After logging in:

1. Click **"Create Project"**
2. Enter a project name (e.g., "My LLM Project")
3. Click **"Create"**

### 4. Get Your API Keys

Langfuse will display your API keys:

- **Public Key:** Used for client-side tracking
- **Secret Key:** Used for server-side operations

**⚠️ IMPORTANT:** Copy and save these keys - they're only shown once!

Store them in a password manager or secure notes.

---

## Managing Langfuse in Portainer

### Viewing Container Status

**Navigation:** Portainer → Containers

Here you can see:

- Container status (running, stopped, healthy)
- CPU and memory usage
- Uptime

### Starting/Stopping Containers

**To stop Langfuse:**

1. Go to **Stacks** → **langfuse**
2. Click **"Stop this stack"**

**To start Langfuse:**

1. Go to **Stacks** → **langfuse**
2. Click **"Start this stack"**

**To restart a single container:**

1. Go to **Containers**
2. Check the box next to the container
3. Click **"Restart"**

### Viewing Logs

**Real-time logs:**

1. **Containers** → Click container name
2. Click **"Logs"** tab
3. Toggle **"Auto-refresh logs"** for live updates

**Search logs:**

1. In the Logs tab, use the search box
2. Search for terms like "error", "warning", "failed"

### Viewing Resource Usage

**Container stats:**

1. **Containers** → Click container name
2. Click **"Stats"** tab
3. View real-time CPU, memory, network usage

**Stack stats:**

1. **Stacks** → Click **"langfuse"**
2. See combined resource usage

### Accessing Container Console

**To run commands inside a container:**

1. **Containers** → Click container name
2. Click **"Console"** tab
3. Click **"Connect"**
4. Choose `/bin/sh` from the dropdown
5. Click **"Connect"** button

Now you have a shell inside the container!

**Example commands:**

```bash
# In langfuse-postgres container
psql -U langfuse -d langfuse

# In langfuse-app container
ls -la /app
```

Type `exit` to leave the console.

### Managing Volumes

**View volumes:**

1. **Volumes** → See all data volumes
2. Click **"langfuse_langfuse-db-data"** to inspect

**Browse volume contents:**

1. Click on volume name
2. Click **"Browse"** button
3. Navigate filesystem (read-only view)

---

## Configuration

### Updating Environment Variables

**To change configuration:**

1. **Stacks** → **langfuse**
2. Click **"Editor"**
3. Modify environment variables in the YAML
4. Click **"Update the stack"**
5. Check **"Re-pull and redeploy"** if updating images
6. Click **"Update"**

### Common Configuration Changes

#### Change Port

```yaml
ports:
  - '3001:3000' # Changes external port to 3001
```

#### Enable Experimental Features

```yaml
environment:
  LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES: true
```

#### Configure SMTP (Email)

```yaml
environment:
  SMTP_CONNECTION_URL: smtp://user:pass@smtp.gmail.com:587
  EMAIL_FROM_NAME: Langfuse
  EMAIL_FROM_ADDRESS: noreply@yourdomain.com
```

#### Memory Limits

```yaml
services:
  langfuse:
    mem_limit: 512m
    mem_reservation: 256m
```

### Environment Variables Reference

| Variable              | Purpose               | Example                               |
| --------------------- | --------------------- | ------------------------------------- |
| `DATABASE_URL`        | PostgreSQL connection | `postgresql://user:pass@host:5432/db` |
| `SALT`                | Encryption salt       | Random 32+ char string                |
| `NEXTAUTH_SECRET`     | Auth secret           | Random 32+ char string                |
| `NEXTAUTH_URL`        | Application URL       | `http://192.168.1.100:3000`           |
| `TELEMETRY_ENABLED`   | Send usage data       | `true` or `false`                     |
| `SMTP_CONNECTION_URL` | Email settings        | SMTP connection string                |

---

## Backup and Maintenance

### Creating Backups via Portainer

#### Method 1: Using Container Console

1. **Containers** → **langfuse-postgres**
2. **Console** → Connect with `/bin/sh`
3. Run backup command:

```bash
pg_dump -U langfuse langfuse > /tmp/backup.sql
```

4. Exit console
5. **Container** → **Stats** → **"Download from container"**
6. Enter path: `/tmp/backup.sql`
7. Click **"Download"**

#### Method 2: Using SSH (Recommended)

From your SSH session:

```bash
# Create backup directory
mkdir -p ~/langfuse-backups

# Backup database
podman exec langfuse-postgres pg_dump -U langfuse langfuse > ~/langfuse-backups/langfuse-$(date +%Y%m%d-%H%M%S).sql

# Compress backup
gzip ~/langfuse-backups/langfuse-*.sql
```

### Automated Backups

Create a backup script:

```bash
nano ~/langfuse-backup.sh
```

Paste:

```bash
#!/bin/bash
BACKUP_DIR=~/langfuse-backups
DATE=$(date +%Y%m%d-%H%M%S)
mkdir -p $BACKUP_DIR

# Backup database
podman exec langfuse-postgres pg_dump -U langfuse langfuse | gzip > $BACKUP_DIR/langfuse-$DATE.sql.gz

# Keep only last 7 backups
ls -t $BACKUP_DIR/langfuse-*.sql.gz | tail -n +8 | xargs -r rm

echo "Backup completed: langfuse-$DATE.sql.gz"
```

Make executable:

```bash
chmod +x ~/langfuse-backup.sh
```

Schedule daily backups:

```bash
crontab -e
```

Add:

```shell
0 2 * * * /home/pi/langfuse-backup.sh >> /home/pi/langfuse-backup.log 2>&1
```

### Restoring from Backup

1. **Stop the stack** (Stacks → langfuse → Stop)

2. Restore via SSH:

```bash
gunzip -c ~/langfuse-backups/langfuse-YYYYMMDD-HHMMSS.sql.gz | podman exec -i langfuse-postgres psql -U langfuse langfuse
```

3. **Start the stack** (Stacks → langfuse → Start)

### Updating Langfuse

**Via Portainer GUI:**

1. **Stacks** → **langfuse**
2. Click **"Editor"**
3. No changes needed to YAML
4. Check **"Re-pull images and redeploy"**
5. Click **"Update the stack"**

Portainer will:

- Pull latest images
- Recreate containers
- Preserve your data

**Check for updates:**

Before updating, check release notes:

- https://github.com/langfuse/langfuse/releases

### Stack Backup and Export

**Export stack configuration:**

1. **Stacks** → **langfuse**
2. Click **"Editor"**
3. Copy the entire YAML
4. Save to a file on your computer

**This is your infrastructure as code!**

### Volume Backup (Full Data Backup)

To backup the entire database volume:

```bash
# Stop the stack first
podman stop langfuse-postgres

# Backup volume
podman volume export langfuse_langfuse-db-data > ~/langfuse-volume-backup.tar

# Restart
podman start langfuse-postgres
```

---

## Monitoring and Logs

### Real-Time Monitoring in Portainer

**Dashboard View:**

1. **Home** → View all environments
2. Shows total containers, images, volumes
3. Quick status at a glance

**Container Stats:**

1. **Containers** → Click container
2. **Stats** tab shows:
   - CPU usage %
   - Memory usage
   - Network I/O
   - Block I/O

### Log Management

**View logs with filters:**

1. **Container** → **Logs**
2. Use search box to filter
3. Select number of lines to show
4. Toggle auto-refresh

**Download logs:**

1. View logs as above
2. Click **"Copy to clipboard"** or manually save
3. Store for analysis

**Common log searches:**

- Search for `error` - Find errors
- Search for `warn` - Find warnings
- Search for `POST` - See API requests
- Search for `failed` - Authentication failures

### Setting Up Alerts (Advanced)

While Portainer CE doesn't have built-in alerting, you can:

1. Monitor logs for specific patterns
2. Use external tools like:
   - Uptime Kuma (Portainer can deploy this too)
   - Prometheus + Alertmanager

### Health Checks

Portainer shows health status:

- **Green dot:** Healthy
- **Yellow dot:** Starting
- **Red dot:** Unhealthy

**Check health details:**

1. **Container** → **Inspect** tab
2. Scroll to "Health" section
3. See last health check results

---

## Troubleshooting

### Common Issues in Portainer

#### 1. Stack Won't Deploy

**Symptom:** Error when clicking "Deploy the stack"

**Solutions:**

1. **Check YAML syntax:**

   - Ensure proper indentation (use spaces, not tabs)
   - Check all quotes are closed
   - Verify no special characters in passwords

2. **Verify environment variables:**

   - All REPLACE*WITH*\* values updated
   - DATABASE_URL password matches POSTGRES_PASSWORD

3. **Check Portainer logs:**
   - Containers → portainer → Logs
   - Look for error messages

#### 2. Containers Keep Restarting

**Symptom:** Container status shows "restarting" or keeps stopping

**Check container logs:**

1. Containers → langfuse-app → Logs
2. Look for error messages

**Common causes:**

- Database not ready (wait 60 seconds)
- Wrong DATABASE_URL
- Invalid SALT or NEXTAUTH_SECRET (must be 32+ chars)
- Out of memory

**Solutions:**

1. Stop and start the stack
2. Wait 2 minutes for database initialization
3. Check logs for specific errors

#### 3. Can't Access Langfuse Web Interface

**Symptom:** Browser shows "Connection refused" or timeout

**Checklist:**

1. ✅ Container is running (Containers → langfuse-app shows green)
2. ✅ Port is correct (3000 by default)
3. ✅ IP address is correct
4. ✅ Firewall allows port 3000

**Test from Pi:**

```bash
curl http://localhost:3000
```

If this works but browser doesn't, it's a network/firewall issue.

**Fix firewall:**

```bash
sudo ufw allow 3000/tcp
sudo ufw reload
```

#### 4. Database Connection Errors

**Symptom:** Logs show "could not connect to database"

**In Portainer:**

1. Containers → langfuse-postgres
2. Check status is "healthy" (green)
3. View logs for errors

**Test database:**

1. langfuse-postgres → Console → Connect
2. Run: `psql -U langfuse -d langfuse`
3. Should connect successfully

**Common fixes:**

- Verify DATABASE_URL format
- Check password matches in both places
- Ensure network is created
- Wait for database health check to pass

#### 5. Out of Memory

**Symptom:** Containers stop randomly, Pi becomes unresponsive

**Check memory in Portainer:**

1. Containers → langfuse-app → Stats
2. Check memory usage

**Add swap file:**

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**Add memory limits in stack:**

```yaml
services:
  langfuse:
    mem_limit: 512m
```

#### 6. Can't Access Portainer

**Symptom:** Portainer web interface won't load

**Solutions:**

```bash
# Check if Portainer is running
podman ps | grep portainer

# If not running, start it
podman start portainer

# View logs
podman logs portainer

# If port 9443 is taken, change it
podman stop portainer
podman rm portainer
# Then recreate with different port (-p 9444:9443)
```

#### 7. Port Already in Use

**Symptom:** Error: "port is already allocated"

**Find what's using the port:**

```bash
sudo lsof -i :3000
```

**Kill the process or change Langfuse port:**

In Portainer:

1. Stacks → langfuse → Editor
2. Change `ports: - "3001:3000"`
3. Update the stack

#### 8. Volume Permission Issues

**Symptom:** Database won't start, permission denied errors

**Solution:**

Ensure `:Z` is on all volume mounts in your compose file:

```yaml
volumes:
  - langfuse-db-data:/var/lib/postgresql/data:Z
```

The `:Z` is critical for SELinux labeling in rootless Podman.

### Getting More Help

**Check logs first:**

1. Portainer → Containers → Click container → Logs
2. Copy relevant error messages

**Useful information when asking for help:**

- Raspberry Pi model and RAM
- Podman version (`podman --version`)
- Full error message from logs
- Steps you've already tried

**Community resources:**

- Langfuse Discord: <https://discord.gg/7NXusRtqYU>
- Portainer Forums: <https://community.portainer.io>
- Podman GitHub: <https://github.com/containers/podman/discussions>

---

## Advanced Portainer Features

### Creating Custom Templates

Save your Langfuse configuration as a template for easy redeployment:

1. **App Templates** → **Custom Templates**
2. Click **"Add Custom Template"**
3. Fill in:
   - Title: "Langfuse Stack"
   - Description: "LLM observability platform"
   - Icon: (optional URL to icon)
   - Platform: Linux
   - Type: Standalone
4. Paste your compose YAML
5. Click **"Create custom template"**

Now you can deploy Langfuse with one click!

### Stack Webhooks (Auto-Deploy on Update)

Set up automatic stack updates when Langfuse releases new versions:

1. **Stacks** → **langfuse**
2. Scroll to **"Webhooks"**
3. Click **"Create webhook"**
4. Click **"Create"**
5. Copy the webhook URL
6. Use with GitHub Actions or CI/CD

### Using Portainer API

Portainer has a REST API for automation:

**Get API token:**

1. **Settings** → **Users**
2. Click on admin user
3. **Access tokens** → **Add access token**
4. Name it, save the token

**Example API calls:**

```bash
# List containers
curl -X GET "https://192.168.1.100:9443/api/endpoints/1/docker/containers/json" \
  -H "X-API-Key: YOUR_TOKEN" --insecure

# Start container
curl -X POST "https://192.168.1.100:9443/api/endpoints/1/docker/containers/langfuse-app/start" \
  -H "X-API-Key: YOUR_TOKEN" --insecure
```

### Multiple Stacks

Deploy multiple Langfuse instances for different purposes:

**Production stack:**

- Name: `langfuse-prod`
- Port: `3000`

**Development stack:**

- Name: `langfuse-dev`
- Port: `3001`
- Different database

Each stack is completely isolated.

### Environment Variables Management

**Use environment variable files:**

1. Create `.env` file locally
2. In Portainer stack editor, click **"Load variables from .env file"**
3. Upload your `.env` file
4. Variables automatically replace placeholders

Example `.env`:

```shell
DB_PASSWORD=your_password_here
SALT=your_salt_here
NEXTAUTH_SECRET=your_secret_here
PI_IP=192.168.1.100
```

Then in YAML:

```yaml
environment:
  POSTGRES_PASSWORD: ${DB_PASSWORD}
  SALT: ${SALT}
  NEXTAUTH_URL: http://${PI_IP}:3000
```

### Resource Quotas

Limit resources per stack:

1. **Stacks** → **langfuse** → **Editor**
2. Add deploy section:

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

### Stack Duplication

Clone your stack for testing:

1. **Stacks** → **langfuse**
2. **Editor** → Copy all YAML
3. **Stacks** → **Add stack**
4. Name: `langfuse-test`
5. Paste YAML, modify ports
6. Deploy

Now you have a test environment!

---

## Performance Optimization Tips

### 1. Use SSD Instead of microSD

Biggest performance improvement for Raspberry Pi:

- Boot from USB SSD
- Reduces I/O bottleneck
- Improves database performance

### 2. Monitor Resource Usage

In Portainer:

1. Home → Check resource gauges
2. Containers → Stats tab for each container
3. Look for memory usage >80% (add swap or upgrade RAM)

### 3. Optimize PostgreSQL

Add to `langfuse-db` environment in stack:

```yaml
environment:
  POSTGRES_SHARED_BUFFERS: 128MB
  POSTGRES_EFFECTIVE_CACHE_SIZE: 256MB
  POSTGRES_WORK_MEM: 4MB
  POSTGRES_MAINTENANCE_WORK_MEM: 64MB
```

### 4. Enable Auto-Prune

Keep system clean:

```bash
# Add to crontab
0 3 * * 0 podman system prune -af >> ~/prune.log 2>&1
```

### 5. Monitor Disk Usage

In SSH session:

```bash
# Check disk space
df -h

# Check container disk usage
podman system df
```

---

## Security Best Practices

### 1. Change Default Passwords

Never use example passwords:

- Generate strong passwords (32+ characters)
- Store in password manager
- Update in stack configuration

### 2. Use Strong Portainer Password

- Minimum 12 characters
- Mix of letters, numbers, symbols
- Enable 2FA if deploying Portainer Business Edition

### 3. Firewall Configuration

```bash
sudo ufw enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow from 192.168.1.0/24 to any port 3000    # Langfuse (local network only)
sudo ufw allow from 192.168.1.0/24 to any port 9443    # Portainer (local network only)
```

### 4. Regular Updates

Check for updates monthly:

1. Stacks → langfuse → Re-pull images
2. Update Raspberry Pi OS: `sudo apt update && sudo apt upgrade`
3. Check Portainer for updates

### 5. Backup Encryption

Encrypt your backups:

```bash
gpg --symmetric ~/langfuse-backups/backup.sql
```

### 6. Network Isolation

Your database is already isolated - it's not exposed externally.

Verify:

```bash
podman port langfuse-postgres
# Should show nothing (no exposed ports)
```

### 7. Use HTTPS (Optional)

For external access, use a reverse proxy:

- Nginx with Let's Encrypt
- Caddy (easier, automatic HTTPS)
- Cloudflare Tunnel (no open ports needed)

---

## Conclusion

Congratulations! You now have:

✅ **Langfuse** running on your Raspberry Pi  
✅ **Portainer** for easy web-based management  
✅ **Podman** for secure, rootless containers  
✅ **Automated backups** configured  
✅ **Monitoring** and logging in place

### Quick Reference

**Access Points:**

- Langfuse: `http://192.168.1.100:3000`
- Portainer: `https://192.168.1.100:9443`

**Common Tasks:**

- Start/Stop: Portainer → Stacks → langfuse
- View Logs: Portainer → Containers → Container → Logs
- Backup: Run `~/langfuse-backup.sh`
- Update: Stacks → langfuse → Re-pull and redeploy

### Next Steps

1. ✅ Configure your LLM applications with Langfuse API keys
2. ✅ Set up external access if needed (reverse proxy)
3. ✅ Test backups and restoration
4. ✅ Monitor resource usage weekly
5. ✅ Join Langfuse Discord for community support

### Additional Resources

- **Langfuse Docs:** https://langfuse.com/docs
- **Portainer Docs:** https://docs.portainer.io
- **Podman Docs:** https://docs.podman.io
- **This Guide:** Keep it handy for reference!

---

**Document Version:** 1.0 (Portainer Edition)  
**Last Updated:** January 2026  
**Tested On:** Raspberry Pi 4 (4GB), Raspberry Pi OS 64-bit, Podman 4.x, Portainer CE

**Author's Note:** This guide prioritizes GUI management through Portainer while maintaining the security benefits of Podman. Most operations can be done through the web interface, making it accessible to users who prefer visual management over command-line operations.
