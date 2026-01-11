# Setting Up Prometheus and Grafana on Portainer

## A Comprehensive Guide to Container Monitoring

---

## Table of Contents

- [Setting Up Prometheus and Grafana on Portainer](#setting-up-prometheus-and-grafana-on-portainer)
  - [A Comprehensive Guide to Container Monitoring](#a-comprehensive-guide-to-container-monitoring)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Prerequisites](#prerequisites)
  - [Architecture Overview](#architecture-overview)
  - [Setting Up Prometheus](#setting-up-prometheus)
    - [Step 1: Create a Docker Network](#step-1-create-a-docker-network)
    - [Step 2: Create Prometheus Configuration](#step-2-create-prometheus-configuration)
    - [Step 3: Deploy Prometheus Container](#step-3-deploy-prometheus-container)
    - [Step 4: Verify Prometheus Installation](#step-4-verify-prometheus-installation)
  - [Setting Up cAdvisor (Optional but Recommended)](#setting-up-cadvisor-optional-but-recommended)
  - [Setting Up Grafana](#setting-up-grafana)
    - [Step 1: Deploy Grafana Container](#step-1-deploy-grafana-container)
    - [Step 2: Access Grafana](#step-2-access-grafana)
    - [Step 3: Add Prometheus as a Data Source](#step-3-add-prometheus-as-a-data-source)
  - [Creating Your First Dashboard](#creating-your-first-dashboard)
    - [Import Pre-built Dashboards](#import-pre-built-dashboards)
    - [Create a Custom Dashboard](#create-a-custom-dashboard)
  - [Advanced Configuration](#advanced-configuration)
    - [Enable Docker Metrics Endpoint](#enable-docker-metrics-endpoint)
    - [Add Node Exporter for Host Metrics](#add-node-exporter-for-host-metrics)
    - [Configure Alerting](#configure-alerting)
  - [Useful PromQL Queries](#useful-promql-queries)
  - [Troubleshooting](#troubleshooting)
    - [Prometheus Cannot Scrape Targets](#prometheus-cannot-scrape-targets)
    - [Grafana Cannot Connect to Prometheus](#grafana-cannot-connect-to-prometheus)
    - [No Metrics Appearing in Dashboards](#no-metrics-appearing-in-dashboards)
    - [Permission Denied on Volume Mounts](#permission-denied-on-volume-mounts)
  - [Best Practices](#best-practices)
    - [Security](#security)
    - [Performance](#performance)
    - [Maintenance](#maintenance)
  - [Alternative: Using Docker Compose](#alternative-using-docker-compose)
  - [Conclusion](#conclusion)

---

## Introduction

This guide provides step-by-step instructions for setting up Prometheus and Grafana using Portainer, a popular container management platform. Prometheus is an open-source monitoring and alerting toolkit, while Grafana is a powerful visualization platform. Together, they form a robust monitoring solution for containerized applications.

By the end of this guide, you will have a fully functional monitoring stack capable of collecting metrics from your Docker containers and displaying them in customizable dashboards.

## Prerequisites

Before beginning this setup, ensure you have the following:

- Portainer installed and running (CE or Business Edition)
- Docker engine version 20.10 or higher
- Administrative access to Portainer
- At least 2GB of available RAM
- Basic understanding of Docker containers and networking

## Architecture Overview

The monitoring stack consists of three main components:

**Prometheus** collects and stores time-series metrics data. It scrapes metrics from configured targets at specified intervals and provides a powerful query language (PromQL) for data analysis.

**Grafana** provides visualization and dashboards. It connects to Prometheus as a data source and allows you to create custom dashboards with graphs, charts, and alerts.

**cAdvisor (optional)** exposes container metrics to Prometheus. While Docker Engine provides some metrics, cAdvisor offers more detailed container-level statistics.

## Setting Up Prometheus

### Step 1: Create a Docker Network

First, create a dedicated Docker network for your monitoring stack. This allows Prometheus and Grafana to communicate securely.

1. In Portainer, navigate to **Networks** (in the left sidebar)
2. Click **Add network**
3. Set the name to `monitoring`
4. Select **bridge** as the driver
5. Click **Create the network**

Alternatively, create it via CLI:

```bash
docker network create monitoring
```

### Step 2: Create Prometheus Configuration

Prometheus requires a configuration file to define what targets to scrape. Create this file before deploying the container.

1. Create a directory for Prometheus configuration on your Docker host:

```bash
mkdir -p /opt/prometheus
```

2. Create the `prometheus.yml` configuration file:

```bash
nano /opt/prometheus/prometheus.yml
```

3. Add the following basic configuration:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'docker'
    static_configs:
      - targets: ['host.docker.internal:9323']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
```

### Step 3: Deploy Prometheus Container

Now deploy Prometheus using Portainer:

1. In Portainer, go to **Containers** and click **Add container**
2. Set the container name to `prometheus`
3. Set the image to `prom/prometheus:latest`
4. Under **Network ports configuration**, map port `9090` to `9090`
5. Under **Advanced container settings > Volumes**, add a bind mount:
   - Container: `/etc/prometheus/prometheus.yml`
   - Host: `/opt/prometheus/prometheus.yml`
6. Add another volume for data persistence:
   - Container: `/prometheus`
   - Host: `/opt/prometheus/data`
7. Under **Network**, select the `monitoring` network
8. Under **Restart policy**, select **Always**
9. Click **Deploy the container**

### Step 4: Verify Prometheus Installation

Once deployed, verify that Prometheus is running:

- Access Prometheus web UI at `http://your-host:9090`
- Navigate to **Status > Targets** to verify scrape targets
- The `prometheus` target should show as **UP**

## Setting Up cAdvisor (Optional but Recommended)

cAdvisor provides detailed container metrics. While optional, it significantly enhances monitoring capabilities.

1. In Portainer, go to **Containers** and click **Add container**
2. Set the container name to `cadvisor`
3. Set the image to `gcr.io/cadvisor/cadvisor:latest`
4. Under **Network ports configuration**, map port `8080` to `8080`
5. Under **Advanced container settings > Volumes**, add these bind mounts:
   - `/:/rootfs:ro`
   - `/var/run:/var/run:ro`
   - `/sys:/sys:ro`
   - `/var/lib/docker/:/var/lib/docker:ro`
   - `/dev/disk/:/dev/disk:ro`
6. Under **Network**, select the `monitoring` network
7. Under **Restart policy**, select **Always**
8. Add this command argument under **Command & logging > Command**:

   ```shell
   --housekeeping_interval=10s
   ```

9. Click **Deploy the container**

## Setting Up Grafana

### Step 1: Deploy Grafana Container

1. In Portainer, go to **Containers** and click **Add container**
2. Set the container name to `grafana`
3. Set the image to `grafana/grafana:latest`
4. Under **Network ports configuration**, map port `3000` to `3000`
5. Under **Advanced container settings > Env** tab, add these environment variables:

   ```shell
   GF_SECURITY_ADMIN_USER=admin
   GF_SECURITY_ADMIN_PASSWORD=admin
   GF_INSTALL_PLUGINS=grafana-clock-panel
   ```

6. Under **Volumes**, create a volume for data persistence:
   - Container: `/var/lib/grafana`
   - Host: `/opt/grafana/data`
7. Under **Network**, select the `monitoring` network
8. Under **Restart policy**, select **Always**
9. Click **Deploy the container**

### Step 2: Access Grafana

After deployment:

- Navigate to `http://your-host:3000`
- Log in with username: `admin` and password: `admin`
- You will be prompted to change the password on first login

### Step 3: Add Prometheus as a Data Source

Configure Grafana to use Prometheus as its data source:

1. In Grafana, click on the menu icon (three lines) in the top left
2. Navigate to **Connections > Data sources**
3. Click **Add data source**
4. Select **Prometheus**
5. Configure the following settings:
   - Name: `Prometheus`
   - URL: `http://prometheus:9090`
6. Scroll down and click **Save & test**
7. You should see a green message confirming the connection

## Creating Your First Dashboard

### Import Pre-built Dashboards

Grafana has a library of community dashboards. Here's how to import a Docker monitoring dashboard:

1. Click on **Dashboards** in the left menu
2. Click **New > Import**
3. For Docker Container & Host Metrics, use dashboard ID: `179`
4. Click **Load**
5. Select **Prometheus** as the data source
6. Click **Import**

Other useful dashboard IDs:

- `193` - Docker and System Monitoring
- `14282` - Docker Container & Host Metrics (detailed)
- `11074` - Node Exporter for Prometheus Dashboard

### Create a Custom Dashboard

To create a custom dashboard:

1. Click **Dashboards > New > New dashboard**
2. Click **Add visualization**
3. Select **Prometheus** as the data source
4. In the query editor, enter a PromQL query, for example:

   ```shell
   rate(container_cpu_usage_seconds_total[5m])
   ```

5. Configure the visualization type (Graph, Gauge, Stat, etc.)
6. Set panel title and other options
7. Click **Apply** to add the panel to your dashboard
8. Click **Save dashboard** and provide a name

## Advanced Configuration

### Enable Docker Metrics Endpoint

To allow Prometheus to scrape Docker daemon metrics:

1. Edit the Docker daemon configuration file:

```bash
sudo nano /etc/docker/daemon.json
```

2. Add or update with:

```json
{
  "metrics-addr": "0.0.0.0:9323",
  "experimental": true
}
```

3. Restart Docker:

```bash
sudo systemctl restart docker
```

4. Restart Portainer after Docker restarts

### Add Node Exporter for Host Metrics

Node Exporter provides detailed host system metrics:

1. In Portainer, create a new container
2. Name: `node-exporter`
3. Image: `prom/node-exporter:latest`
4. Port mapping: `9100:9100`
5. Network: `monitoring`
6. Add volume mounts:
   - `/proc:/host/proc:ro`
   - `/sys:/host/sys:ro`
   - `/:/rootfs:ro`
7. Add command arguments:

   ```shell
   --path.procfs=/host/proc
   --path.sysfs=/host/sys
   --collector.filesystem.mount-points-exclude='^/(sys|proc|dev|host|etc)($$|/)'
   ```

8. Deploy the container
9. Update `prometheus.yml` to add node-exporter scrape target:

```yaml
- job_name: 'node'
  static_configs:
    - targets: ['node-exporter:9100']
```

10. Restart Prometheus to apply the configuration

### Configure Alerting

Grafana supports alerting on metric thresholds. To set up a basic alert:

1. Open or create a dashboard panel
2. Click the panel title and select **Edit**
3. In the panel editor, go to the **Alert** tab
4. Click **Create alert rule from this panel**
5. Configure alert conditions (e.g., CPU usage above 80%)
6. Set up notification channels (email, Slack, etc.) in **Alerting > Contact points**

## Useful PromQL Queries

Here are some common queries for monitoring Docker containers:

| Metric                 | PromQL Query                                                                                       | Description                                 |
| ---------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| Container CPU Usage    | `rate(container_cpu_usage_seconds_total{name!=""}[5m])`                                            | CPU usage rate per container over 5 minutes |
| Container Memory Usage | `container_memory_usage_bytes{name!=""}`                                                           | Current memory usage by container           |
| Container Network I/O  | `rate(container_network_receive_bytes_total[5m])`                                                  | Network receive rate                        |
| Container Disk I/O     | `rate(container_fs_writes_bytes_total[5m])`                                                        | Filesystem write rate                       |
| Container Count        | `count(container_last_seen)`                                                                       | Total number of containers                  |
| Node CPU Usage         | `100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)`                 | Host CPU usage percentage                   |
| Node Memory Usage      | `(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100` | Host memory usage percentage                |
| Disk Space Used        | `100 - ((node_filesystem_avail_bytes * 100) / node_filesystem_size_bytes)`                         | Filesystem usage percentage                 |

## Troubleshooting

### Prometheus Cannot Scrape Targets

**Problem:** Targets show as DOWN in Prometheus Status page

**Solutions:**

- Verify all containers are on the same Docker network (`monitoring`)
- Check that container names in `prometheus.yml` match actual container names
- Ensure ports are correctly exposed on target containers
- Review Prometheus logs for connection errors

### Grafana Cannot Connect to Prometheus

**Problem:** Data source test fails in Grafana

**Solutions:**

- Confirm both containers are on the `monitoring` network
- Use container name (`prometheus`) instead of localhost or IP
- Verify Prometheus is accessible at `http://prometheus:9090` from within the network
- Check Grafana logs for detailed error messages

### No Metrics Appearing in Dashboards

**Problem:** Dashboards show no data or empty graphs

**Solutions:**

- Wait a few minutes for metrics to be collected (initial scrape interval)
- Verify time range in Grafana dashboard (top right) covers current time
- Check that scrape targets are UP in Prometheus Status page
- Test PromQL queries directly in Prometheus web UI
- Ensure correct data source is selected in dashboard

### Permission Denied on Volume Mounts

**Problem:** Containers fail to start due to permission errors

**Solutions:**

- Ensure host directories exist before mounting
- Set appropriate permissions on host directories:

```bash
sudo chmod -R 755 /opt/prometheus
sudo chmod -R 755 /opt/grafana
```

- For Grafana specifically, you may need to set ownership:

```bash
sudo chown -R 472:472 /opt/grafana/data
```

## Best Practices

### Security

- Change default Grafana admin password immediately
- Use a reverse proxy (nginx, Traefik) for HTTPS access
- Restrict network access to monitoring ports (9090, 3000)
- Consider using Grafana authentication integration (OAuth, LDAP)
- Regularly update container images for security patches

### Performance

- Adjust scrape intervals based on your needs (15s is aggressive, 60s is often sufficient)
- Configure Prometheus retention period to manage disk usage:

  ```shell
  --storage.tsdb.retention.time=15d
  ```

- Use recording rules for frequently used complex queries
- Limit dashboard auto-refresh rates to reduce system load

### Maintenance

- Implement regular backups of Grafana dashboards and Prometheus data
- Monitor Prometheus and Grafana resource usage
- Document custom dashboards and alert rules
- Use version control for configuration files
- Set up monitoring for the monitoring stack itself

## Alternative: Using Docker Compose

While this guide focuses on Portainer GUI deployment, you can also use Docker Compose. Here's a complete `docker-compose.yml` file:

```yaml
version: '3.8'

networks:
  monitoring:
    driver: bridge

volumes:
  prometheus-data:
  grafana-data:

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    restart: always
    ports:
      - 9090:9090
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    restart: always
    ports:
      - 3000:3000
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - monitoring

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: cadvisor
    restart: always
    ports:
      - 8080:8080
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
    networks:
      - monitoring

  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    restart: always
    ports:
      - 9100:9100
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - monitoring
```

## Conclusion

You now have a complete monitoring solution with Prometheus and Grafana running on Portainer. This setup provides comprehensive visibility into your Docker containers and host system metrics.

The modular nature of this stack allows you to easily extend it with additional exporters and data sources as your monitoring needs grow. Consider exploring Prometheus Alertmanager for advanced alerting capabilities, and investigate additional Grafana plugins to enhance your dashboards.

Remember to regularly review and update your monitoring configuration to ensure it continues to meet your operational requirements.
