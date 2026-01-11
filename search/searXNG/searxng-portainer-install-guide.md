# Installing SearXNG on Raspberry Pi with Portainer

A complete guide to setting up SearXNG (privacy-respecting metasearch engine) on your Raspberry Pi using Portainer.

## Prerequisites

- Portainer already installed on your Raspberry Pi
- Basic understanding of Docker and Docker Compose
- SSH access to your Raspberry Pi

---

## Installation Steps

### 1. Prepare the Directory Structure

SSH into your Raspberry Pi and create directories for SearXNG:

```bash
mkdir -p ~/searxng/config
cd ~/searxng
```

### 2. Create the Settings File

Create a basic `settings.yml` in the config directory:

```bash
nano config/settings.yml
```

Add this minimal configuration:

```yaml
use_default_settings: true
server:
  secret_key: 'CHANGE_THIS_TO_RANDOM_STRING' # Generate with: openssl rand -hex 32
  limiter: false
  image_proxy: true
search:
  safe_search: 0
  autocomplete: 'google'
engines:
  - name: google
    disabled: false
```

**Generate a secret key:**

```bash
openssl rand -hex 32
```

Copy the output and replace `CHANGE_THIS_TO_RANDOM_STRING` in your settings.yml file.

### 3. Create Docker Compose Stack in Portainer

1. Log into Portainer web interface
2. Navigate to **Stacks** → **Add stack**
3. Name it `searxng`
4. Choose **Web editor** and paste the following:

#### Basic Configuration

```yaml
version: '3.7'

services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    restart: unless-stopped
    ports:
      - '8080:8080' # Change external port if needed
    volumes:
      - ./config:/etc/searxng:rw
    environment:
      - SEARXNG_BASE_URL=http://YOUR_PI_IP:8080/ # Change to your Pi's IP
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - SETGID
      - SETUID
    logging:
      driver: 'json-file'
      options:
        max-size: '1m'
        max-file: '1'
```

**Important:** Replace `YOUR_PI_IP` with your Raspberry Pi's actual IP address.

### 4. Raspberry Pi Specific Adjustments

Since you're running on ARM architecture, consider these optimizations:

#### For Limited RAM (Optional)

Add memory limits to prevent system crashes:

```yaml
deploy:
  resources:
    limits:
      memory: 512M
```

Add this under the `searxng` service configuration.

### 5. Deploy the Stack

1. Scroll down in Portainer
2. Click **Deploy the stack**
3. Wait for the container to start (check the status in the Containers view)

### 6. Access SearXNG

Open your browser and navigate to:

```text
http://YOUR_PI_IP:8080
```

For example: `http://192.168.1.100:8080`

---

## Optional Enhancements

### Add Redis for Better Performance

Redis provides caching which significantly improves performance. Update your stack to include Redis:

```yaml
version: '3.7'

services:
  redis:
    image: redis:alpine
    container_name: searxng-redis
    restart: unless-stopped
    command: redis-server --save 30 1 --loglevel warning
    volumes:
      - redis-data:/data

  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    restart: unless-stopped
    ports:
      - '8080:8080'
    volumes:
      - ./config:/etc/searxng:rw
    environment:
      - SEARXNG_BASE_URL=http://YOUR_PI_IP:8080/
      - SEARXNG_REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - SETGID
      - SETUID
    logging:
      driver: 'json-file'
      options:
        max-size: '1m'
        max-file: '1'

volumes:
  redis-data:
```

### Advanced Settings Configuration

Create a more detailed `settings.yml` for production use:

```yaml
use_default_settings: true

server:
  secret_key: 'YOUR_SECRET_KEY_HERE'
  limiter: false
  image_proxy: true
  bind_address: '0.0.0.0'
  port: 8080

ui:
  static_use_hash: true
  default_locale: 'en'
  theme_args:
    simple_style: auto

search:
  safe_search: 0
  autocomplete: 'google'
  autocomplete_min: 4
  default_lang: 'en'
  max_page: 0

outgoing:
  request_timeout: 3.0
  max_request_timeout: 10.0
  pool_connections: 100
  pool_maxsize: 20
  enable_http2: true

engines:
  - name: google
    disabled: false

  - name: duckduckgo
    disabled: false

  - name: bing
    disabled: false

  - name: wikipedia
    disabled: false

  - name: github
    disabled: false
```

### Set Up Reverse Proxy with HTTPS (Recommended for External Access)

If you want secure HTTPS access, consider adding Nginx Proxy Manager or Traefik:

#### Option 1: Nginx Proxy Manager

1. Install Nginx Proxy Manager in Portainer
2. Create a proxy host pointing to `searxng:8080`
3. Enable SSL with Let's Encrypt

#### Option 2: Traefik

Add labels to your SearXNG service for automatic SSL:

```yaml
labels:
  - 'traefik.enable=true'
  - 'traefik.http.routers.searxng.rule=Host(`search.yourdomain.com`)'
  - 'traefik.http.routers.searxng.entrypoints=websecure'
  - 'traefik.http.routers.searxng.tls.certresolver=letsencrypt'
```

---

## Troubleshooting

### Container Won't Start

1. **Check logs in Portainer:**

   - Click the container → **Logs** tab
   - Look for error messages

2. **Verify permissions:**

   ```bash
   chmod -R 755 ~/searxng/config
   chown -R 1000:1000 ~/searxng/config
   ```

3. **Ensure config file is valid:**

   ```bash
   cat ~/searxng/config/settings.yml
   ```

### Slow Performance

1. **Reduce active engines:**
   Edit `settings.yml` and disable unused engines:

   ```yaml
   engines:
     - name: google
       disabled: false
     - name: bing
       disabled: true # Disable unused engines
   ```

2. **Add Redis caching** (see Optional Enhancements above)

3. **Check Raspberry Pi temperature:**

   ```bash
   vcgencmd measure_temp
   ```

   Ensure adequate cooling if temperature > 70°C

4. **Optimize timeout settings:**

   ```yaml
   outgoing:
     request_timeout: 2.0 # Reduce from 3.0
   ```

### Can't Access from Other Devices

1. **Check firewall rules:**

   ```bash
   sudo ufw status
   sudo ufw allow 8080/tcp
   ```

2. **Verify you're using correct IP:**

   ```bash
   hostname -I
   ```

3. **Test locally first:**

   ```bash
   curl http://localhost:8080
   ```

### High Memory Usage

1. **Add memory limits to docker-compose:**

   ```yaml
   deploy:
     resources:
       limits:
         memory: 512M
       reservations:
         memory: 256M
   ```

2. **Reduce pool connections:**

   ```yaml
   outgoing:
     pool_connections: 50 # Reduce from 100
     pool_maxsize: 10 # Reduce from 20
   ```

### Search Results Not Loading

1. **Check internet connectivity:**

   ```bash
   ping 8.8.8.8
   ```

2. **Verify DNS resolution:**

   ```bash
   nslookup google.com
   ```

3. **Test individual engines:**
   - Go to SearXNG preferences
   - Disable all engines except one
   - Test if that engine works

---

## Maintenance

### Updating SearXNG

1. In Portainer, go to **Stacks**
2. Select your `searxng` stack
3. Click **Editor**
4. Change the image tag if needed (or keep `latest`)
5. Click **Update the stack**
6. Check the box **Re-pull image and redeploy**
7. Click **Update**

### Backup Configuration

```bash
# Backup settings
cp ~/searxng/config/settings.yml ~/searxng/config/settings.yml.backup

# Backup entire directory
tar -czf searxng-backup-$(date +%Y%m%d).tar.gz ~/searxng/
```

### Monitor Logs

```bash
# View real-time logs
docker logs -f searxng

# View last 100 lines
docker logs --tail 100 searxng
```

---

## Additional Resources

- **Official SearXNG Documentation:** <https://docs.searxng.org/>
- **SearXNG GitHub:** <https://github.com/searxng/searxng>
- **SearXNG Settings Reference:** <https://docs.searxng.org/admin/settings/index.html>
- **Community Instances:** <https://searx.space/>

---

## Security Considerations

1. **Change the default secret key** - Never use the example key in production
2. **Use HTTPS** - Set up a reverse proxy with SSL for external access
3. **Rate limiting** - Enable limiter in settings.yml if exposed to internet:

   ```yaml
   server:
     limiter: true
   ```

4. **Keep updated** - Regularly update the SearXNG container
5. **Firewall** - Only expose necessary ports

---

## Performance Tips for Raspberry Pi

1. **Use Redis** - Significantly improves response times
2. **Limit concurrent searches** - Reduce pool connections
3. **Disable unused engines** - Only enable search engines you actually use
4. **Use SSD instead of SD card** - If possible, run from USB SSD
5. **Overclock (optional)** - If you have proper cooling
6. **Use lite OS** - Raspberry Pi OS Lite uses less resources

---

_Last Updated: January 2026_
