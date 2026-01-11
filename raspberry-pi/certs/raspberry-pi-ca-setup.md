# Setting Up a Local CA Certificate on Raspberry Pi

This guide explains how to create and install a local Certificate Authority (CA) certificate on your Raspberry Pi. This is useful for trusting self-signed certificates or creating your own internal PKI.

## Option 1: Create Your Own CA Certificate

### 1. Generate the CA private key and certificate

```bash
# Create a directory for your CA
mkdir -p ~/ca-setup
cd ~/ca-setup

# Generate CA private key
openssl genrsa -out ca-key.pem 4096

# Generate CA certificate (valid for 10 years)
openssl req -new -x509 -days 3650 -key ca-key.pem -out ca-cert.pem
```

You'll be prompted to enter information for the certificate (Country, State, Organization, etc.).

### 2. Install the CA certificate system-wide

```bash
# Copy the certificate to the trusted certificates directory
sudo cp ca-cert.pem /usr/local/share/ca-certificates/my-local-ca.crt

# Update the certificate store
sudo update-ca-certificates
```

You should see output like: `1 added, 0 removed; done.`

## Option 2: Install an Existing CA Certificate

If you already have a CA certificate file:

```bash
# Copy your existing CA certificate
sudo cp /path/to/your-ca-cert.crt /usr/local/share/ca-certificates/

# Update the certificate store
sudo update-ca-certificates
```

## Verify Installation

```bash
# Check if your certificate is in the trusted store
ls /etc/ssl/certs/ | grep -i "your-cert-name"

# Test with OpenSSL
openssl verify /usr/local/share/ca-certificates/my-local-ca.crt
```

## For Use with Python/pip

If you need to use the CA cert with Python applications:

```bash
# Find the CA bundle location
python3 -c "import certifi; print(certifi.where())"

# Or set environment variable
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

## For Use with curl/wget

These tools should automatically use the system certificate store after running `update-ca-certificates`.

## Important Security Notes

- Keep your CA private key (`ca-key.pem`) secure and backed up
- Only install CA certificates you trust
- Never share your CA private key
- Store the private key in a secure location with restricted permissions:

  ```bash
  chmod 600 ~/ca-setup/ca-key.pem
  ```

## Common Use Cases

### Creating Server Certificates Signed by Your CA

Once you have your CA set up, you can create server certificates:

```bash
# Generate server private key
openssl genrsa -out server-key.pem 2048

# Create certificate signing request (CSR)
openssl req -new -key server-key.pem -out server.csr

# Sign the certificate with your CA
openssl x509 -req -in server.csr -CA ca-cert.pem -CAkey ca-key.pem \
  -CAcreateserial -out server-cert.pem -days 365
```

### Configuring Applications

Different applications may need different configurations to use your CA certificate. Consult the specific application's documentation for certificate trust configuration.
