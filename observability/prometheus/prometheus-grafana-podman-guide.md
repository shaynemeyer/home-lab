# Setting Up Prometheus and Grafana on Raspberry Pi with Portainer and Podman

## A Comprehensive Guide to Container Monitoring on Raspberry Pi OS

---

## Table of Contents

- [Setting Up Prometheus and Grafana on Raspberry Pi with Portainer and Podman](#setting-up-prometheus-and-grafana-on-raspberry-pi-with-portainer-and-podman)
  - [A Comprehensive Guide to Container Monitoring on Raspberry Pi OS](#a-comprehensive-guide-to-container-monitoring-on-raspberry-pi-os)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Prerequisites](#prerequisites)
    - [Raspberry Pi Specific Considerations](#raspberry-pi-specific-considerations)
  - [Architecture Overview](#architecture-overview)
  - [Food for Thought: Raspberry Pi Specific Considerations](#food-for-thought-raspberry-pi-specific-considerations)
    - [ARM Architecture](#arm-architecture)
    - [Storage Recommendations](#storage-recommendations)
    - [Performance Tuning](#performance-tuning)
    - [Model-Specific Notes](#model-specific-notes)
    - [Cooling](#cooling)
  - [Podman vs Docker Considerations](#podman-vs-docker-considerations)
    - [Volume Paths and Permissions](#volume-paths-and-permissions)
    - [Networking](#networking)
    - [What Stays the Same](#what-stays-the-same)
  - [Setting Up Prometheus](#setting-up-prometheus)
    - [Step 1: Create a Podman Network](#step-1-create-a-podman-network)
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
    - [Enable Podman Metrics Endpoint](#enable-podman-metrics-endpoint)
    - [Add Node Exporter for Host Metrics](#add-node-exporter-for-host-metrics)
    - [Configure Alerting](#configure-alerting)
  - [Useful PromQL Queries](#useful-promql-queries)
  - [Troubleshooting](#troubleshooting)
    - [Prometheus Cannot Scrape Targets](#prometheus-cannot-scrape-targets)
    - [Grafana Cannot Connect to Prometheus](#grafana-cannot-connect-to-prometheus)
    - [No Metrics Appearing in Dashboards](#no-metrics-appearing-in-dashboards)
    - [Permission Denied on Volume Mounts](#permission-denied-on-volume-mounts)
    - [cAdvisor Not Collecting Podman Metrics](#cadvisor-not-collecting-podman-metrics)
    - [Container Won't Start](#container-wont-start)
    - [Raspberry Pi Specific Issues](#raspberry-pi-specific-issues)
  - [Best Practices](#best-practices)
    - [Security](#security)
    - [Performance](#performance)
    - [Raspberry Pi Specific Best Practices](#raspberry-pi-specific-best-practices)
    - [Maintenance](#maintenance)
    - [Portainer-Specific Best Practices](#portainer-specific-best-practices)
  - [Alternative: Using Portainer Stacks](#alternative-using-portainer-stacks)
    - [Using Portainer Stacks](#using-portainer-stacks)
    - [Advantages of Using Stacks](#advantages-of-using-stacks)
  - [Conclusion](#conclusion)

---

## Introduction

This guide provides step-by-step instructions for setting up Prometheus and Grafana on a **Raspberry Pi** running Raspberry Pi OS, using Portainer as your container management interface with Podman as the underlying container engine. Prometheus is an open-source monitoring and alerting toolkit, while Grafana is a powerful visualization platform. Together, they form a robust monitoring solution for containerized applications.

All images used in this guide are **ARM-compatible** and tested to work on Raspberry Pi hardware. You'll use Portainer's web interface for most operations. However, there are a few Podman-specific and Raspberry Pi-specific considerations around volume paths, networking, permissions, and performance that this guide addresses.

By the end of this guide, you will have a fully functional monitoring stack capable of collecting metrics from your Podman containers and displaying them in customizable dashboards, all running on your Raspberry Pi.

## Prerequisites

Before beginning this setup, ensure you have the following:

- **Raspberry Pi** (Pi 4 or Pi 5 recommended for best performance)
- **Raspberry Pi OS** (64-bit recommended)
- Portainer installed and already configured to manage Podman containers
- Podman version 4.0 or higher (as the container engine)
- Administrative access to Portainer web interface
- At least 2GB of available RAM (4GB+ recommended)
- SD card with at least 16GB free space (or external storage for better performance)
- Basic understanding of container networking

**Note:** This guide assumes your Portainer instance is already successfully managing Podman on your Raspberry Pi. If you haven't set this up yet, ensure Portainer can connect to the Podman socket before proceeding.

### Raspberry Pi Specific Considerations

- All container images used in this guide support **ARM64 architecture**
- Performance may vary based on your Pi model (Pi 4/5 recommended)
- Consider using external SSD/USB storage instead of SD card for better I/O performance
- Monitor temperature and consider adding cooling if running multiple monitoring containers

## Architecture Overview

The monitoring stack consists of three main components:

**Prometheus** collects and stores time-series metrics data. It scrapes metrics from configured targets at specified intervals and provides a powerful query language (PromQL) for data analysis.

**Grafana** provides visualization and dashboards. It connects to Prometheus as a data source and allows you to create custom dashboards with graphs, charts, and alerts.

**cAdvisor (optional)** exposes container metrics to Prometheus. While Podman provides some metrics, cAdvisor offers more detailed container-level statistics.

## Food for Thought: Raspberry Pi Specific Considerations

### ARM Architecture

All container images in this guide are multi-architecture and include ARM64 builds. The images will automatically pull the correct ARM version for your Raspberry Pi.

### Storage Recommendations

- **SD Card**: Works but may be slow for database writes
- **Recommended**: Use USB 3.0 SSD or NVMe HAT for better performance
- **Data Path**: Store Prometheus and Grafana data on faster storage if available

### Performance Tuning

For Raspberry Pi, we'll adjust some default settings:

- Increase Prometheus scrape interval to reduce CPU load (30s instead of 15s)
- Reduce Prometheus retention to save disk space (7 days instead of 15)
- Limit memory usage for each container
- Disable some resource-intensive cAdvisor metrics

### Model-Specific Notes

- **Pi 3 or earlier**: May struggle with full stack, consider running only Prometheus + Grafana
- **Pi 4 (4GB+)**: Can run full monitoring stack comfortably
- **Pi 5**: Excellent performance, can handle additional exporters

### Cooling

Monitor your Pi's temperature, especially when running multiple containers:

```bash
vcgencmd measure_temp
```

Consider adding a heatsink or fan if temperatures exceed 70°C under load.

## Podman vs Docker Considerations

When using Portainer with Podman as the backend instead of Docker, there are a few key differences to be aware of. Most operations in Portainer will look identical, but these differences affect volume paths and configuration:

### Volume Paths and Permissions

- **Rootless Podman**: Volumes are typically stored in `~/.local/share/containers/storage/volumes/`
- **Rootful Podman**: Volumes are in `/var/lib/containers/storage/volumes/`
- **SELinux**: Volume mounts may require `:Z` suffix (handled automatically by Portainer in most cases)
- When creating bind mounts in Portainer, use the appropriate path for your Podman setup

### Networking

- The default bridge network works the same way
- Use `host.containers.internal` instead of `host.docker.internal` to reach the host from containers
- Container-to-container communication on the same network works identically

### What Stays the Same

- Portainer's web interface and workflows
- Container management (create, start, stop, delete)
- Network creation and management
- All the steps in this guide will use Portainer's GUI the same way you're used to

## Setting Up Prometheus

### Step 1: Create a Podman Network

First, create a dedicated network for your monitoring stack using Portainer. This allows Prometheus and Grafana to communicate securely.

1. In Portainer, navigate to **Networks** (in the left sidebar)
2. Click **Add network**
3. Set the name to `monitoring`
4. Select **bridge** as the driver
5. Click **Create the network**

That's it! Portainer will create the network using Podman in the background.

### Step 2: Create Prometheus Configuration

Prometheus requires a configuration file to define what targets to scrape. You'll need to create this file on your Raspberry Pi (the host where Podman is running).

**Access your Raspberry Pi via SSH or terminal**, then:

1. Create a directory for Prometheus configuration:

```bash
# For rootless Podman (most common on Raspberry Pi OS)
mkdir -p ~/podman/prometheus

# For rootful Podman (if running as root)
sudo mkdir -p /opt/prometheus
```

2. Create the `prometheus.yml` configuration file:

```bash
# For rootless Podman
nano ~/podman/prometheus/prometheus.yml

# For rootful Podman
sudo nano /opt/prometheus/prometheus.yml
```

3. Add the following **Raspberry Pi optimized** configuration:

```yaml
global:
  scrape_interval: 30s # Increased from 15s to reduce Pi CPU load
  evaluation_interval: 30s
  scrape_timeout: 10s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'raspberry-pi'
    static_configs:
      - targets: ['node-exporter:9100']
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        replacement: 'raspberry-pi'
```

**Note:** The 30-second scrape interval reduces CPU usage on the Raspberry Pi while still providing good monitoring granularity.

**Important:** Note the path you created (`~/podman/prometheus/prometheus.yml` or `/opt/prometheus/prometheus.yml`) - you'll need this when setting up the container in Portainer.

### Step 3: Deploy Prometheus Container

Now deploy Prometheus using Portainer's interface with Raspberry Pi optimized settings:

1. In Portainer, go to **Containers** and click **Add container**
2. Set the container name to `prometheus`
3. Set the image to `prom/prometheus:latest` (will auto-pull ARM64 version)
4. Under **Network ports configuration**, click "publish a new network port"
   - Host: `9090`
   - Container: `9090`
5. Under **Advanced container settings > Volumes** tab, click "map additional volume"
   - **For the config file:**
     - Container: `/etc/prometheus/prometheus.yml`
     - Host: `/home/pi/podman/prometheus/prometheus.yml` (adjust if your username isn't "pi")
   - Click "+ map additional volume" again
   - **For data persistence:**
     - Container: `/prometheus`
     - Host: `/home/pi/podman/prometheus/data` (or path to external storage if using SSD)
6. Under **Advanced container settings > Command & logging** tab, add these command arguments:

   ```shell
   --storage.tsdb.retention.time=7d --storage.tsdb.path=/prometheus --config.file=/etc/prometheus/prometheus.yml
   ```

   - This limits data retention to 7 days to save disk space on SD card

7. Under **Advanced container settings > Resources** tab (optional but recommended):
   - Memory limit: `512MB` (prevents Prometheus from using too much RAM)
   - Memory reservation: `256MB`
8. Under **Network** tab, select the `monitoring` network from the dropdown
9. Under **Restart policy** tab, select **Always**
10. Click **Deploy the container**

**Note:** Portainer will automatically pull the ARM64 version of the Prometheus image for your Raspberry Pi.

### Step 4: Verify Prometheus Installation

Once deployed, verify that Prometheus is running:

1. In Portainer, go to **Containers** and check that the `prometheus` container shows as "running"
2. Click on the container name to view details and check the logs for any errors
3. Access the Prometheus web UI at `http://your-host-ip:9090`
4. In Prometheus, navigate to **Status > Targets** to verify scrape targets
5. The `prometheus` target should show as **UP**

If you see errors, click on the container in Portainer and check the **Logs** tab for details.

## Setting Up cAdvisor (Optional but Recommended)

cAdvisor provides detailed container metrics. While optional, it significantly enhances monitoring capabilities. We'll use a lighter configuration for Raspberry Pi.

1. In Portainer, go to **Containers** and click **Add container**
2. Set the container name to `cadvisor`
3. Set the image to `gcr.io/cadvisor/cadvisor:latest` (ARM64 compatible)
4. Under **Network ports configuration**, publish port:
   - Host: `8080`
   - Container: `8080`
5. Under **Advanced container settings > Volumes** tab, add these bind mounts (click "map additional volume" for each):
   - `/:/rootfs` (read-only: check the box)
   - `/var/run:/var/run` (read-only: check the box)
   - `/sys:/sys` (read-only: check the box)
   - `/var/lib/containers:/var/lib/containers` (read-only: check the box)
     - _For rootless Podman: `/home/pi/.local/share/containers:/var/lib/containers`_
6. Under **Network** tab, select the `monitoring` network
7. Under **Advanced container settings > Resources** tab:
   - Memory limit: `256MB`
   - Memory reservation: `128MB`
8. Under **Restart policy** tab, select **Always**
9. Under **Command & logging** tab, in the **Command** field, add (optimized for Pi):

   ```shell
   --housekeeping_interval=30s --docker_only=false --disable_metrics=percpu,sched,tcp,udp,disk,diskIO,accelerator,hugetlb,referenced_memory,cpu_topology,resctrl
   ```

   - This disables resource-intensive metrics to reduce CPU load

10. Click **Deploy the container**

**Raspberry Pi Note:** The reduced metrics collection significantly lowers CPU usage while still providing essential container monitoring data.

## Setting Up Grafana

### Step 1: Deploy Grafana Container

1. In Portainer, go to **Containers** and click **Add container**
2. Set the container name to `grafana`
3. Set the image to `grafana/grafana:latest` (ARM64 compatible)
4. Under **Network ports configuration**, publish port:
   - Host: `3000`
   - Container: `3000`
5. Under **Advanced container settings > Env** tab, click "add environment variable" for each:
   - Name: `GF_SECURITY_ADMIN_USER` Value: `admin`
   - Name: `GF_SECURITY_ADMIN_PASSWORD` Value: `admin` (change after first login!)
   - Name: `GF_INSTALL_PLUGINS` Value: `grafana-clock-panel`
   - Name: `GF_SERVER_ROOT_URL` Value: `http://raspberrypi.local:3000` (adjust to your Pi's hostname)
6. Under **Volumes** tab, map additional volume:
   - Container: `/var/lib/grafana`
   - Host: `/home/pi/podman/grafana/data` (or path to external storage if using SSD)
7. Under **Advanced container settings > Resources** tab:
   - Memory limit: `512MB`
   - Memory reservation: `256MB`
8. Under **Network** tab, select the `monitoring` network
9. Under **Restart policy** tab, select **Always**
10. Click **Deploy the container**

**If you encounter permission errors**, create the directory on your Pi first:

```bash
# For rootless Podman
mkdir -p ~/podman/grafana/data
chmod 755 ~/podman/grafana/data
```

**Raspberry Pi Note:** Grafana may take 1-2 minutes to start on first run as it initializes the database.

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

Grafana has a library of community dashboards. Here are dashboards that work great on Raspberry Pi:

1. Click on **Dashboards** in the left menu
2. Click **New > Import**
3. **For Raspberry Pi Monitoring**, use dashboard ID: `10578`
   - This dashboard is specifically designed for Raspberry Pi with node-exporter
4. Click **Load**
5. Select **Prometheus** as the data source
6. Click **Import**

**Other useful dashboard IDs for Raspberry Pi:**

- `1860` - Node Exporter Full (comprehensive system metrics)
- `179` - Container & Host Metrics (works with Podman)
- `11074` - Node Exporter for Prometheus Dashboard
- `15798` - Podman Dashboard
- `893` - Raspberry Pi specific metrics (if using additional exporters)

**Raspberry Pi Specific Metrics to Monitor:**

- CPU Temperature (important for Pi health!)
- SD Card I/O
- Memory usage (Pis have limited RAM)
- Network throughput
- Container resource consumption

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

### Enable Podman Metrics Endpoint

To get Podman-specific metrics, you can use the Podman exporter. This provides detailed information about Podman containers that complements cAdvisor.

1. In Portainer, go to **Containers** and click **Add container**
2. Set the container name to `podman-exporter`
3. Set the image to `quay.io/navidys/prometheus-podman-exporter:latest`
4. Under **Network ports configuration**, publish port:
   - Host: `9882`
   - Container: `9882`
5. Under **Advanced container settings > Volumes** tab, add:
   - Container: `/var/run/podman/podman.sock`
   - Host: `/run/podman/podman.sock` (rootful) or `/run/user/YOUR_UID/podman/podman.sock` (rootless)
     - _To find YOUR_UID for rootless: run `echo $UID` on your Podman host_
6. Under **Network** tab, select the `monitoring` network
7. Under **Restart policy** tab, select **Always**
8. Click **Deploy the container**

9. Update your `prometheus.yml` file on the host to add the Podman exporter:

```yaml
- job_name: 'podman-exporter'
  static_configs:
    - targets: ['podman-exporter:9882']
```

10. Restart the Prometheus container in Portainer (select the container and click **Restart**)

### Add Node Exporter for Host Metrics

Node Exporter provides detailed host system metrics, including Raspberry Pi specific hardware monitoring:

1. In Portainer, create a new container (click **Add container**)
2. Name: `node-exporter`
3. Image: `prom/node-exporter:latest` (ARM64 compatible)
4. Under **Network ports configuration**, publish port:
   - Host: `9100`
   - Container: `9100`
5. Under **Network** tab, select `monitoring`
6. Under **Volumes** tab, add these bind mounts (all read-only):
   - `/proc:/host/proc` (check read-only)
   - `/sys:/sys` (check read-only)
   - `/:/rootfs` (check read-only)
7. Under **Command & logging** tab, in the **Command** field, add:

   ```shell
   --path.procfs=/host/proc --path.sysfs=/host/sys --collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/) --collector.textfile.directory=/var/lib/node_exporter/textfile_collector
   ```

8. Under **Advanced container settings > Resources** tab:
   - Memory limit: `128MB`
   - Memory reservation: `64MB`
9. Under **Restart policy** tab, select **Always**
10. Click **Deploy the container**

11. Update `prometheus.yml` on your Pi to add the node-exporter scrape target:

```yaml
- job_name: 'node'
  static_configs:
    - targets: ['node-exporter:9100']
```

12. Restart the Prometheus container in Portainer (select it and click **Restart**)

**Raspberry Pi Temperature Monitoring:**

To add CPU temperature monitoring to Node Exporter, create a simple script on your Pi:

```bash
# Create script directory
mkdir -p ~/podman/node-exporter-textfile

# Create temperature monitoring script
cat > ~/check_temp.sh << 'EOF'
#!/bin/bash
TEMP=$(vcgencmd measure_temp | sed 's/temp=//' | sed 's/°C//')
echo "# HELP raspberry_pi_temp_celsius Raspberry Pi CPU Temperature" > ~/podman/node-exporter-textfile/rpi_temp.prom
echo "# TYPE raspberry_pi_temp_celsius gauge" >> ~/podman/node-exporter-textfile/rpi_temp.prom
echo "raspberry_pi_temp_celsius $TEMP" >> ~/podman/node-exporter-textfile/rpi_temp.prom
EOF

chmod +x ~/check_temp.sh

# Add to crontab to run every minute
(crontab -l 2>/dev/null; echo "* * * * * ~/check_temp.sh") | crontab -
```

Then add this volume mount to your node-exporter container in Portainer:

- Container: `/var/lib/node_exporter/textfile_collector`
- Host: `/home/pi/podman/node-exporter-textfile`

### Configure Alerting

Grafana supports alerting on metric thresholds. To set up a basic alert:

1. Open or create a dashboard panel
2. Click the panel title and select **Edit**
3. In the panel editor, go to the **Alert** tab
4. Click **Create alert rule from this panel**
5. Configure alert conditions (e.g., CPU usage above 80%)
6. Set up notification channels (email, Slack, etc.) in **Alerting > Contact points**

## Useful PromQL Queries

Here are some common queries for monitoring Podman containers:

| Metric                 | PromQL Query                                                                                       | Description                                 |
| ---------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| Container CPU Usage    | `rate(container_cpu_usage_seconds_total{name!=""}[5m])`                                            | CPU usage rate per container over 5 minutes |
| Container Memory Usage | `container_memory_usage_bytes{name!=""}`                                                           | Current memory usage by container           |
| Container Network I/O  | `rate(container_network_receive_bytes_total[5m])`                                                  | Network receive rate                        |
| Container Disk I/O     | `rate(container_fs_writes_bytes_total[5m])`                                                        | Filesystem write rate                       |
| Container Count        | `count(container_last_seen)`                                                                       | Total number of containers                  |
| Podman Container Count | `podman_container_info`                                                                            | Podman-specific container count             |
| Node CPU Usage         | `100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)`                 | Host CPU usage percentage                   |
| Node Memory Usage      | `(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100` | Host memory usage percentage                |
| Disk Space Used        | `100 - ((node_filesystem_avail_bytes * 100) / node_filesystem_size_bytes)`                         | Filesystem usage percentage                 |

## Troubleshooting

### Prometheus Cannot Scrape Targets

**Problem:** Targets show as DOWN in Prometheus Status page

**Solutions:**

- In Portainer, verify all containers are on the same network (`monitoring`):
  - Click on each container and check the **Network** tab
- Check that container names in `prometheus.yml` match the actual container names in Portainer
- Verify ports are correctly exposed:
  - In Portainer, click on each target container and check **Port configuration**
- Check Prometheus logs:
  - In Portainer, select the Prometheus container and click **Logs** tab
- Test network connectivity between containers:
  - In Portainer, select Prometheus container
  - Click **Console** and select `/bin/sh`
  - Try: `wget -O- http://cadvisor:8080` or `ping cadvisor`

### Grafana Cannot Connect to Prometheus

**Problem:** Data source test fails in Grafana

**Solutions:**

- Verify both containers are on the `monitoring` network in Portainer
- Use container name (`prometheus`) not localhost or IP in Grafana data source URL
- In Portainer, check Grafana logs for detailed error messages (click container > **Logs**)
- Test from Grafana console:
  - In Portainer, select Grafana container > **Console** > `/bin/sh`
  - Run: `wget -O- http://prometheus:9090/-/healthy`

### No Metrics Appearing in Dashboards

**Problem:** Dashboards show no data or empty graphs

**Solutions:**

- Wait 2-3 minutes for initial metrics collection
- Check time range in Grafana dashboard (top right) covers current time
- Verify scrape targets are UP in Prometheus **Status > Targets** page
- Test PromQL queries directly in Prometheus web UI (Graph tab)
- In Portainer, check that Prometheus container has been running long enough to collect data
- Check Prometheus storage:
  - In Portainer, select Prometheus > **Stats** to see if disk usage is growing

### Permission Denied on Volume Mounts

**Problem:** Containers fail to start due to permission errors

**Solutions:**

- Check container logs in Portainer (select container > **Logs** tab)
- Ensure host directories exist before creating containers
- For rootless Podman, ensure directories are owned by your user:

```bash
ls -la ~/podman/prometheus/
ls -la ~/podman/grafana/
```

- Create directories with correct permissions:

```bash
mkdir -p ~/podman/prometheus/data ~/podman/grafana/data
chmod 755 ~/podman/prometheus/data ~/podman/grafana/data
```

- If still having issues with Grafana specifically:

```bash
# For rootless Podman, Grafana runs as UID 472
mkdir -p ~/podman/grafana/data
chmod 777 ~/podman/grafana/data  # Temporary to test
```

### cAdvisor Not Collecting Podman Metrics

**Problem:** cAdvisor shows no container metrics

**Solutions:**

- Verify cAdvisor has correct volume mounts in Portainer:
  - Click on cAdvisor container > **Inspect**
  - Check Mounts section for `/var/lib/containers`
- For rootless Podman, ensure you're mounting the correct path:
  - Should be: `/home/YOUR_USER/.local/share/containers:/var/lib/containers`
- Check cAdvisor logs in Portainer
- Verify cAdvisor can access the web UI at `http://your-host:8080`

### Container Won't Start

**Problem:** Container shows as "Exited" or error state in Portainer

**Solutions:**

- Click on the container in Portainer and check the **Logs** tab for error messages
- Common issues:
  - **Port already in use**: Another container or process is using the port. Check with different port
  - **Volume path doesn't exist**: Create the directory on the host first
  - **Image pull failed**: Check internet connectivity and image name spelling
  - **ARM compatibility**: Ensure you're using multi-arch images (all images in this guide are ARM64 compatible)
- Try **Recreate** button in Portainer to redeploy with same settings

### Raspberry Pi Specific Issues

**Problem:** High CPU temperature (>70°C) or throttling

**Solutions:**

- Check temperature: `vcgencmd measure_temp`
- Check for throttling: `vcgencmd get_throttled` (0x0 is good, anything else indicates throttling)
- Add cooling (heatsink or fan)
- Reduce container resource usage:
  - Increase scrape intervals in `prometheus.yml`
  - Disable more cAdvisor metrics
  - Limit concurrent container startup
- Check `dmesg` for thermal warnings

**Problem:** SD card performance issues or corruption

**Solutions:**

- Move data directories to USB SSD:

  ```bash
  # Example: Moving to USB drive mounted at /mnt/usb
  sudo mkdir -p /mnt/usb/monitoring/{prometheus,grafana}
  sudo chown pi:pi /mnt/usb/monitoring -R
  ```

  Then update volume paths in Portainer to `/mnt/usb/monitoring/prometheus/data` etc.

- Use high-quality SD card (Class 10 or better)
- Monitor SD card health periodically
- Consider using SSD boot if on Pi 4/5

**Problem:** Containers running slow or system unresponsive

**Solutions:**

- Check available memory: `free -h`
- Review container resource limits in Portainer
- Stop non-essential containers
- Consider running minimal stack (only Prometheus + Grafana, skip cAdvisor)
- Check for swap usage: `swapon -s`
- Increase swap if needed (for monitoring containers):

  ```bash
  sudo dphys-swapfile swapoff
  sudo nano /etc/dphys-swapfile  # Change CONF_SWAPSIZE=2048
  sudo dphys-swapfile setup
  sudo dphys-swapfile swapon
  ```

**Problem:** Network connectivity issues between containers

**Solutions:**

- Verify network exists: In Portainer, go to **Networks** and check `monitoring` is listed
- Restart Podman network:
  ```bash
  podman network rm monitoring
  podman network create monitoring
  ```
- Check firewall rules: `sudo iptables -L`
- Ensure containers are all on the same network in Portainer

## Best Practices

### Security

- Change default Grafana admin password immediately
- Use rootless Podman when possible for enhanced security
- Use a reverse proxy (nginx, Traefik) for HTTPS access
- Restrict network access to monitoring ports (9090, 3000)
- Consider using Grafana authentication integration (OAuth, LDAP)
- Regularly update container images: `podman auto-update`
- Use SELinux in enforcing mode with proper labels

### Performance

- **For Raspberry Pi**, use longer scrape intervals (30-60s instead of 15s)
- Configure Prometheus retention based on available storage:
  - SD Card: `--storage.tsdb.retention.time=3d` to `7d`
  - External SSD: `--storage.tsdb.retention.time=15d` to `30d`
- Use recording rules for frequently used complex queries
- Limit dashboard auto-refresh rates (30s or 1min minimum)
- Set memory limits on all containers to prevent OOM on Raspberry Pi
- Monitor resource usage through Portainer's Stats tab
- Consider disabling unnecessary cAdvisor metrics (already done in this guide)

### Raspberry Pi Specific Best Practices

**Storage:**

- Use external USB 3.0 SSD for better performance and longevity
- If using SD card, use high-quality cards (Samsung EVO, SanDisk Extreme)
- Monitor SD card health regularly
- Keep at least 20% free space on storage device

**Cooling:**

- Monitor temperature: `watch vcgencmd measure_temp`
- Target: Keep under 70°C under load
- Add passive cooling (heatsink) minimum
- Consider active cooling (fan) for 24/7 operation
- Ensure proper ventilation around the Pi

**Power:**

- Use official Raspberry Pi power supply or quality 5V 3A+ adapter
- Undervoltage causes instability and SD card corruption
- Check for power issues: `vcgencmd get_throttled`
- Avoid USB-powered setups for production monitoring

**Resource Management:**

- Set resource limits on all containers
- Enable swap for additional memory headroom (2GB recommended)
- Limit concurrent container operations in Portainer
- Schedule updates during low-usage periods
- Use `htop` or Grafana to monitor system resources

**Networking:**

- Use wired Ethernet for reliability
- If using WiFi, ensure strong signal
- Disable unnecessary network services
- Monitor network I/O in dashboards

**Updates:**

- Keep Raspberry Pi OS updated: `sudo apt update && sudo apt upgrade`
- Update container images regularly through Portainer
- Update Podman: `sudo apt update && sudo apt install podman`
- Test updates on non-production first

**Backup Strategy:**

- Backup Grafana dashboards regularly (export as JSON)
- Backup Prometheus data directory if needed
- Consider full SD card image backup monthly
- Store backups off-device
- Monitor Podman resource usage: `podman stats`

### Maintenance

- Regularly backup Grafana dashboards:
  - In Grafana, go to **Dashboards** > select dashboard > **Settings** > **JSON Model** > copy and save
- Backup Prometheus data directory on your host
- Monitor resource usage in Portainer:
  - Click on **Home** to see resource usage across all containers
  - Click individual containers to see detailed stats
- Document custom dashboards and alert rules
- Keep configuration files (`prometheus.yml`) in version control
- Update container images regularly:
  - In Portainer, select container > **Recreate** with "Pull latest image" checked

### Portainer-Specific Best Practices

- Use Portainer **Stacks** feature for managing multiple related containers together
- Set up **Backup** schedules for important data volumes in Portainer
- Use Portainer's **Environment variables** management for sensitive data
- Monitor containers through Portainer's built-in stats and logs
- Create **Templates** in Portainer for quick redeployment of your monitoring stack

## Alternative: Using Portainer Stacks

Since you're using Portainer, the best way to deploy this entire monitoring stack at once is through **Portainer Stacks** (Portainer's docker-compose equivalent). This allows you to manage all containers as a single unit.

### Using Portainer Stacks

1. First, create your `prometheus.yml` configuration file on the host (as described in Step 2 earlier)

2. In Portainer, go to **Stacks** in the left menu

3. Click **Add stack**

4. Give it a name like `monitoring-stack`

5. In the **Web editor**, paste the following Raspberry Pi optimized compose file:

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
      - '9090:9090'
    volumes:
      - /home/pi/podman/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:Z
      - prometheus-data:/prometheus:Z
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=7d'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    networks:
      - monitoring
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    restart: always
    ports:
      - '3000:3000'
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_SERVER_ROOT_URL=http://raspberrypi.local:3000
    volumes:
      - grafana-data:/var/lib/grafana:Z
    networks:
      - monitoring
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: cadvisor
    restart: always
    ports:
      - '8080:8080'
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/containers/:/var/lib/containers:ro
      - /home/pi/.local/share/containers:/home/pi/.local/share/containers:ro
    networks:
      - monitoring
    command:
      - '--housekeeping_interval=30s'
      - '--docker_only=false'
      - '--disable_metrics=percpu,sched,tcp,udp,disk,diskIO,accelerator,hugetlb,referenced_memory,cpu_topology,resctrl'
    deploy:
      resources:
        limits:
          memory: 256M
        reservations:
          memory: 128M

  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    restart: always
    ports:
      - '9100:9100'
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    networks:
      - monitoring
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    deploy:
      resources:
        limits:
          memory: 128M
        reservations:
          memory: 64M

  # Optional: Podman-specific metrics
  podman-exporter:
    image: quay.io/navidys/prometheus-podman-exporter:latest
    container_name: podman-exporter
    restart: always
    ports:
      - '9882:9882'
    volumes:
      # For rootless Podman - adjust UID (1000 is typical for 'pi' user)
      - /run/user/1000/podman/podman.sock:/var/run/podman/podman.sock:Z
    networks:
      - monitoring
    deploy:
      resources:
        limits:
          memory: 128M
```

6. **Important Raspberry Pi Adjustments**:

   - Update `/home/pi/podman/prometheus/prometheus.yml` if your username isn't "pi"
   - For rootful Podman, change paths to `/opt/prometheus/prometheus.yml`
   - Update `/run/user/1000/podman/podman.sock` - replace `1000` with your user ID (run `echo $UID`)
   - For rootful Podman socket, use: `/run/podman/podman.sock`
   - If using external storage, update volume paths to your SSD mount point

7. Click **Deploy the stack**

**Note:** This compose file includes:

- Memory limits optimized for Raspberry Pi
- 7-day retention for Prometheus (saves SD card space)
- Reduced metrics collection from cAdvisor
- 30-second intervals to reduce CPU load
- All ARM64-compatible images

Portainer will create all containers, networks, and volumes at once. You can manage the entire stack from the **Stacks** page - start, stop, or remove everything together.

### Advantages of Using Stacks

- Deploy/remove all containers with one click
- Easy to version control (export the compose file)
- Update the entire stack at once
- View all container logs in one place
- Cleaner than managing individual containers

## Conclusion

You now have a complete monitoring solution with Prometheus and Grafana running on your Raspberry Pi, managed through Portainer with Podman as the container engine. This setup provides comprehensive visibility into your Podman containers and host system metrics, all optimized for Raspberry Pi's ARM architecture and resource constraints.

The combination of Raspberry Pi, Portainer, and Podman offers several advantages:

- **Cost-Effective**: Low-power monitoring solution (typically <10W)
- **Easy Management**: Portainer's GUI makes container management intuitive
- **Enhanced Security**: Podman's daemonless, rootless architecture
- **ARM Optimized**: All images tested and configured for Raspberry Pi
- **Low Power Consumption**: Perfect for 24/7 monitoring operations

**Your monitoring stack includes:**

- Prometheus for metrics collection and storage
- Grafana for visualization and dashboards
- cAdvisor for detailed container metrics
- Node Exporter for system and hardware monitoring
- Optional Podman Exporter for Podman-specific metrics

You can extend this stack with additional exporters and data sources as your monitoring needs grow. Consider:

- **Alertmanager** for advanced alerting and notification routing
- **Loki** for log aggregation alongside metrics
- **Additional exporters** for application-specific metrics (MySQL, PostgreSQL, etc.)
- **Custom scripts** for Raspberry Pi specific monitoring (GPU temp, voltage, etc.)

**Raspberry Pi Monitoring Reminders:**

- Monitor CPU temperature regularly (keep under 70°C)
- Check for throttling: `vcgencmd get_throttled`
- Use external SSD for better performance and longevity
- Keep storage usage under 80%
- Update OS and containers regularly
- Implement proper cooling for 24/7 operation
- Backup your dashboards and configuration

With Portainer managing your Podman containers on Raspberry Pi, you have a powerful, secure, energy-efficient, and easy-to-use monitoring infrastructure that can run reliably for years with minimal maintenance.

**Next Steps:**

1. Customize your Grafana dashboards for your specific needs
2. Set up alerting for critical metrics (temperature, disk space, container health)
3. Create automated backups of your Grafana configuration
4. Monitor your monitoring stack to ensure it stays healthy
5. Share your setup with the Raspberry Pi and Homelab communities!

Happy monitoring! 🍓📊
