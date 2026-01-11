# Setting Up SSH from Mac to Raspberry Pi

This guide explains how to set up SSH access from your Mac to your Raspberry Pi for remote administration and development.

## On the Raspberry Pi

### 1. Enable SSH

**If you have a monitor connected:**

- Go to Raspberry Pi Configuration → Interfaces → Enable SSH

**Without a monitor:**

- Create an empty file named `ssh` (no extension) in the boot partition of your SD card

### 2. Find Your Pi's IP Address

```bash
hostname -I
```

## On Your Mac

### 1. Test the Connection

```bash
ssh pi@<raspberry-pi-ip-address>
```

Default password is usually "raspberry" (unless you've changed it).

### 2. Set Up SSH Keys for Passwordless Login (Recommended)

```bash
# Generate a key if you don't have one
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy it to your Pi
ssh-copy-id pi@<raspberry-pi-ip-address>
```

### 3. Optional - Add SSH Config Entry for Easier Access

Edit `~/.ssh/config` and add:

```shell
Host raspi
    HostName <raspberry-pi-ip-address>
    User pi
    IdentityFile ~/.ssh/id_ed25519
```

Then you can simply type:

```bash
ssh raspi
```

## Tips

### Using mDNS Instead of IP Address

If your Pi's IP address changes (common with DHCP), you can use:

```bash
ssh pi@raspberrypi.local
```

This works if mDNS (Bonjour) is functioning on your network.

### Setting a Static IP Address

To avoid IP address changes, consider setting a static IP on your Pi:

1. Edit the dhcpcd configuration:

   ```bash
   sudo nano /etc/dhcpcd.conf
   ```

2. Add these lines (adjust for your network):

   ```shell
   interface eth0
   static ip_address=192.168.1.100/24
   static routers=192.168.1.1
   static domain_name_servers=192.168.1.1 8.8.8.8
   ```

3. Restart networking:

   ```bash
   sudo systemctl restart dhcpcd
   ```

## Security Best Practices

- Change the default password immediately
- Use SSH keys instead of passwords
- Consider disabling password authentication in `/etc/ssh/sshd_config`:

  ```bash
  PasswordAuthentication no
  ```

- Keep your Pi updated:

  ```bash
  sudo apt update && sudo apt upgrade -y
  ```

## Troubleshooting

**Connection refused:**

- Check if SSH is enabled on the Pi
- Verify the IP address is correct
- Check firewall settings

**Host key verification failed:**

- Remove the old key: `ssh-keygen -R <raspberry-pi-ip-address>`
- Then reconnect

**Permission denied:**

- Verify username is correct (default is `pi`)
- Check if password is correct
- Ensure SSH keys are properly copied
