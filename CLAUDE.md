# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a documentation repository for home lab setup and configuration. It contains comprehensive guides, tutorials, and troubleshooting documentation for self-hosting various services on Raspberry Pi and other home lab hardware using container technologies.

## Repository Structure

The repository is organized by service categories:

- **`automation/`** - Workflow automation tools (n8n)
- **`containers/`** - Container runtime documentation (Podman troubleshooting)
- **`LangFuse/`** - LLM observability platform setup guides
- **`llms/ollama/`** - Local LLM setup and vector database tutorials (LanceDB)
- **`observability/`** - Monitoring and metrics (Prometheus, Grafana)
- **`raspberry-pi/`** - Raspberry Pi specific configuration (SSH, certificates)
- **`search/`** - Self-hosted search engines (SearXNG)
- **`traefik/`** - Reverse proxy configuration

## Technology Stack

The documentation consistently uses this technology stack:

### Container Runtime
- **Podman** (rootless, daemonless alternative to Docker)
- **Portainer** (web-based container management GUI)
- **podman-compose** for multi-container deployments

### Infrastructure
- **Raspberry Pi 4/5** as primary deployment target (ARM64)
- **Ubuntu/Debian/Raspberry Pi OS** as base operating systems
- **Traefik** as reverse proxy for service routing

### Deployment Patterns
- **Rootless containers** for enhanced security
- **Systemd Quadlets** for service auto-start
- **Named volumes** for persistent storage
- **Custom Podman networks** for service isolation
- **Cloudflare Tunnel** for secure external access (no port forwarding)

### Common Services
- PostgreSQL databases for persistence
- Google OAuth integration patterns
- Let's Encrypt/self-signed certificates for HTTPS
- Prometheus + Grafana for monitoring

## Key Configuration Patterns

### Podman Socket Setup (Required for Portainer)
```bash
systemctl --user enable --now podman.socket
loginctl enable-linger $USER
```

### Port Binding for Rootless Containers
```bash
echo "net.ipv4.ip_unprivileged_port_start=80" | sudo tee /etc/sysctl.d/99-podman-ports.conf
sudo sysctl -p /etc/sysctl.d/99-podman-ports.conf
```

### Raspberry Pi Boot Parameters for Containers
In `/boot/firmware/cmdline.txt` (single line):
```
cgroup_enable=cpuset cgroup_enable=memory cgroup_memory=1
```

### Volume Mounting in Podman
Always append `:Z` for SELinux labeling in rootless mode:
```yaml
volumes:
  - volume-name:/path:Z
```

## Common Commands

### Container Management
```bash
# List running containers
podman ps

# View container logs
podman logs -f <container-name>

# Restart container
podman restart <container-name>

# Execute command in container
podman exec -it <container-name> bash
```

### Service Management (Systemd User Services)
```bash
# Check service status
systemctl --user status <service-name>

# Restart service
systemctl --user restart <service-name>

# View service logs
journalctl --user -u <service-name> -f

# Reload systemd configuration
systemctl --user daemon-reload
```

### Network Management
```bash
# Create network
podman network create <network-name>

# List networks
podman network ls

# Inspect network
podman network inspect <network-name>
```

### Portainer Management
Access Portainer at: `https://<pi-ip>:9443`

Common operations via Portainer:
- Deploy stacks using docker-compose YAML
- View container logs and stats
- Manage volumes and networks
- Update containers by re-pulling images

## Documentation Standards

When working with this repository:

1. **Guide Format**: All guides follow comprehensive tutorial format with:
   - Table of contents
   - Prerequisites section
   - Step-by-step instructions
   - Troubleshooting section
   - Command reference

2. **Command Examples**: Include both manual and automated approaches:
   - Direct Podman commands
   - docker-compose YAML configurations
   - Systemd Quadlet configurations

3. **Architecture Diagrams**: Some guides include Mermaid diagrams for visualization

4. **Version Information**: Guides include tested versions and last update dates

5. **Platform Specificity**: Clearly indicate ARM64 vs x86_64 specific instructions

## Troubleshooting References

Common issues documented in the repository:

1. **Podman cgroup memory errors** - See `containers/podman/troubleshooting/podman-cgroup-memory-troubleshooting.md`
   - Enable memory cgroup in boot parameters
   - Configure systemd delegation for rootless Podman

2. **Port binding issues** - Rootless Podman cannot bind to ports < 1024 by default
   - Lower unprivileged port start threshold via sysctl

3. **Socket connection issues** - Portainer requires Podman socket
   - Enable and start podman.socket via systemd
   - Map socket at `/run/user/${UID}/podman/podman.sock`

4. **Container auto-start** - Requires lingering enabled
   - Use `loginctl enable-linger $USER`

## Service-Specific Notes

### Traefik
- Uses Docker labels for automatic service discovery
- Requires external network: `traefik-public`
- Configuration split between `traefik.yml` and labels
- Dashboard exposed on port 8080

### Prometheus + Grafana
- cAdvisor provides container metrics
- Node Exporter provides host metrics
- All services on dedicated `monitoring` network
- Grafana accessible on port 3000, Prometheus on 9090

### LangFuse
- Requires PostgreSQL database
- Needs strong secrets for SALT and NEXTAUTH_SECRET
- Generate secrets with: `openssl rand -base64 32`
- Accessible on port 3000

### n8n
- Uses PostgreSQL for workflow persistence
- Supports Cloudflare Tunnel for external access
- Google OAuth configuration required for Google services
- Deployed using Quadlets for auto-start

## Security Considerations

1. **Rootless containers**: All guides emphasize rootless Podman deployment
2. **Secrets management**: Generate strong random secrets for all services
3. **Network isolation**: Services use dedicated networks
4. **Firewall rules**: Restrict access to local network or via Cloudflare Tunnel
5. **HTTPS**: Use Traefik with Let's Encrypt or Cloudflare for SSL/TLS

## File Naming Conventions

- Guide filenames use kebab-case with descriptive names
- Suffix indicates deployment method: `-portainer-guide.md`, `-podman-guide.md`
- Troubleshooting files prefixed with issue description

## When Adding New Documentation

1. Follow existing guide structure with comprehensive sections
2. Include both Portainer GUI and CLI approaches when applicable
3. Document tested versions and hardware
4. Add troubleshooting section for common issues
5. Include command reference for quick lookup
6. Specify resource requirements (RAM, storage, CPU)
7. Document backup and maintenance procedures
