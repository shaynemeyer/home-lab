# Podman Deployment Error: memory.max - Troubleshooting Guide

## Problem

**Error Message:**

```text
Failed to deploy a stack: compose up operation failed:
Error response from daemon: crun: opening file `memory.max` for writing:
No such file or directory: OCI runtime attempted to invoke a command that was not found
```

## Environment

- **Device:** Raspberry Pi 4
- **OS:** Debian (ARM64)
- **Podman Version:** 4.3.1
- **Cgroup Version:** v2
- **Container Runtime:** crun

## Root Cause

The memory cgroup controller was **disabled by default** in the Raspberry Pi firmware boot parameters. The error occurred because Podman couldn't set memory limits without the memory controller available.

## Diagnosis Steps

### 1. Check Cgroup Version

```bash
mount | grep cgroup
# Output: cgroup2 on /sys/fs/cgroup type cgroup2 (rw,nosuid,nodev,noexec,relatime,nsdelegate,memory_recursiveprot)
```

### 2. Check Available Controllers at System Level

```bash
cat /sys/fs/cgroup/cgroup.controllers
# Initial output: cpuset cpu io pids
# Missing: memory
```

### 3. Check Podman's View of Controllers

```bash
podman info | grep -A 7 cgroupControllers
# Initial output showed only:
# - cpu
# - pids
```

### 4. Check Kernel Boot Parameters

```bash
cat /proc/cmdline
# Found: cgroup_disable=memory (added by Pi firmware)
```

## Solution

### Step 1: Enable Memory Controller in Boot Parameters

Edit the Raspberry Pi boot configuration:

```bash
sudo nano /boot/firmware/cmdline.txt
# OR on some systems:
sudo nano /boot/cmdline.txt
```

Add to the **end of the single line** (don't create new lines):

```shell
cgroup_enable=memory cgroup_memory=1 swapaccount=1
```

**Example final line:**

```shell
console=tty1 root=PARTUUID=b917f34b-02 rootfstype=ext4 fsck.repair=yes rootwait quiet splash plymouth.ignore-serial-consoles cfg80211.ieee80211_regdom=US cgroup_enable=memory cgroup_memory=1 swapaccount=1
```

### Step 2: Enable Cgroup Delegation for Rootless Podman

Create systemd delegation configuration:

```bash
sudo mkdir -p /etc/systemd/system/user@.service.d/

sudo tee /etc/systemd/system/user@.service.d/delegate.conf > /dev/null <<'EOF'
[Service]
Delegate=cpu cpuset io memory pids
EOF

sudo systemctl daemon-reload
```

### Step 3: Reboot

```bash
sudo reboot
```

### Step 4: Verify After Reboot

```bash
# Check kernel parameters
cat /proc/cmdline | grep cgroup

# Verify all controllers are available
cat /sys/fs/cgroup/cgroup.controllers
# Should show: cpuset cpu io memory pids

# Verify Podman sees all controllers
podman info | grep -A 7 cgroupControllers
# Should show all 5 controllers: cpuset, cpu, io, memory, pids
```

### Step 5: Restart User Session (if needed)

If Podman doesn't see all controllers after reboot:

```bash
# Restart your user service
sudo systemctl restart user@$(id -u).service

# OR logout completely and log back in
```

## Final Verification

```bash
podman info | grep -A 7 cgroupControllers
```

**Expected output:**

```yaml
cgroupControllers:
  - cpuset
  - cpu
  - io
  - memory
  - pids
cgroupManager: systemd
cgroupVersion: v2
```

## Deploy Your Stack

```bash
podman-compose up
# or
podman compose up
```

## Key Takeaways

1. **Raspberry Pi Firmware** adds `cgroup_disable=memory` by default on some configurations
2. **Parameter order matters** - adding `cgroup_enable=memory` at the end of cmdline.txt overrides the earlier disable
3. **Rootless Podman** requires proper systemd delegation for cgroup controllers
4. **User session restart** is necessary for delegation changes to take effect

## Additional Notes

### Why Raspberry Pi Disables Memory Cgroup

The Pi firmware disables the memory cgroup controller to reduce overhead on resource-constrained systems. However, for container workloads, you need this controller enabled.

### Alternative: Run as Root

If you don't need rootless containers, you can run as root instead:

```bash
sudo podman-compose up
```

This bypasses the delegation requirements, but rootless is generally preferred for security.

## References

- [Podman Documentation](https://docs.podman.io/)
- [Cgroups v2 Documentation](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html)
- [Raspberry Pi Boot Configuration](https://www.raspberrypi.com/documentation/computers/config_txt.html)
