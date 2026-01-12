# Home Lab Documentation

My notes and learning from setting up a self-hosted home lab infrastructure. This repository contains comprehensive guides for deploying containerized services on Raspberry Pi and other home lab hardware.

## Overview

This repository documents my journey building a production-ready home lab using modern containerization technologies. All guides emphasize security through rootless containers, ease of management via Portainer, and reliable auto-start capabilities using systemd.

## Technology Stack

- **Container Runtime**: Podman (rootless, daemonless)
- **Management UI**: Portainer
- **Hardware**: Raspberry Pi 4/5 (ARM64)
- **Operating Systems**: Raspberry Pi OS, Ubuntu, Debian
- **Reverse Proxy**: Traefik
- **Monitoring**: Prometheus + Grafana
- **External Access**: Cloudflare Tunnel

## Services Documented

### Automation
- **[n8n](automation/n8n/n8n-setup-guide.md)** - Workflow automation platform with PostgreSQL, Podman Quadlets, and Cloudflare Tunnel integration

### Observability & Monitoring
- **[Prometheus + Grafana](observability/prometheus/prometheus-grafana-portainer-guide.md)** - Complete monitoring stack with cAdvisor and Node Exporter
- **[LangFuse](LangFuse/langfuse-raspberrypi-portainer-guide.md)** - LLM observability platform for tracking AI/ML workloads

### Infrastructure
- **[Traefik](traefik/traefik-portainer-setup-guide.md)** - Reverse proxy with automatic service discovery and SSL/TLS
- **[SearXNG](search/searXNG/searxng-portainer-install-guide.md)** - Self-hosted metasearch engine

### LLM & AI
- **[LanceDB + Ollama](llms/ollama/lancedb/lancedb_ollama_tutorial.md)** - Local RAG (Retrieval Augmented Generation) system
- **[Vector Databases Deep Dive](llms/ollama/lancedb/vector_databases_deep_dive.md)** - Understanding vector search
- **[Embeddings Deep Dive](llms/ollama/lancedb/embeddings_deep_dive.md)** - How embeddings work

### Raspberry Pi Configuration
- **[SSH Setup](raspberry-pi/ssh/SSH_Setup_Mac_to_Raspberry_Pi.md)** - Secure SSH configuration from Mac to Pi
- **[CA Certificate Setup](raspberry-pi/certs/raspberry-pi-ca-setup.md)** - Local Certificate Authority for internal PKI

### Troubleshooting
- **[Podman cgroup Memory Issues](containers/podman/troubleshooting/podman-cgroup-memory-troubleshooting.md)** - Fix memory controller errors on Raspberry Pi

## Quick Start

### Prerequisites
- Raspberry Pi 4 or 5 (4GB+ RAM recommended)
- Raspberry Pi OS (64-bit) or Ubuntu Server
- Basic command line knowledge
- SSH access to your Pi

### Essential Setup Steps

1. **Enable container features** in `/boot/firmware/cmdline.txt`:
   ```
   cgroup_enable=cpuset cgroup_enable=memory cgroup_memory=1
   ```

2. **Install Podman**:
   ```bash
   sudo apt update
   sudo apt install -y podman podman-compose
   ```

3. **Configure rootless Podman**:
   ```bash
   systemctl --user enable --now podman.socket
   loginctl enable-linger $USER
   ```

4. **Allow binding to privileged ports**:
   ```bash
   echo "net.ipv4.ip_unprivileged_port_start=80" | sudo tee /etc/sysctl.d/99-podman-ports.conf
   sudo sysctl -p /etc/sysctl.d/99-podman-ports.conf
   ```

5. **Deploy Portainer** for web-based management:
   ```bash
   podman run -d -p 9443:9443 --name portainer --restart=always \
     -v /run/user/$(id -u)/podman/podman.sock:/var/run/docker.sock:Z \
     -v portainer_data:/data:Z \
     docker.io/portainer/portainer-ce:latest
   ```
   Access at `https://<pi-ip>:9443`

6. **Follow service-specific guides** from the documentation above

## Repository Structure

```
.
├── automation/          # Workflow automation (n8n)
├── containers/          # Container runtime documentation
├── LangFuse/           # LLM observability platform
├── llms/ollama/        # Local LLM and vector DB tutorials
├── observability/      # Monitoring stack (Prometheus, Grafana)
├── raspberry-pi/       # Pi-specific configuration
├── search/             # Self-hosted search (SearXNG)
└── traefik/            # Reverse proxy setup
```

## Key Features of These Guides

- **Security First**: All guides use rootless Podman for enhanced security
- **Production Ready**: Includes auto-start, health checks, and persistent storage
- **Comprehensive**: Prerequisites, setup, troubleshooting, and maintenance
- **Dual Approach**: Both Portainer GUI and CLI instructions provided
- **Well Tested**: All guides tested on Raspberry Pi 4/5 hardware

## Common Patterns

### Networking
All services use dedicated Podman networks for isolation:
```bash
podman network create <service>-network
```

### Auto-Start with Systemd Quadlets
Service definitions in `~/.config/containers/systemd/`:
- Network: `<name>.network`
- Container: `<name>.container`

### Volume Persistence
Named volumes with SELinux labeling for rootless mode:
```yaml
volumes:
  - volume-name:/container/path:Z
```

### Secret Generation
Strong secrets for all services:
```bash
openssl rand -base64 32
```

## Contributing

These are personal notes, but feel free to open issues if you find errors or have suggestions for improvements.

## Resources

- [Podman Documentation](https://docs.podman.io/)
- [Portainer Documentation](https://docs.portainer.io/)
- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)
- [Traefik Documentation](https://doc.traefik.io/traefik/)

## License

Documentation is provided as-is for educational purposes. Individual services have their own licenses.
