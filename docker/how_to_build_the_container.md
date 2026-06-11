# Adaptive MFD Container Replication Guide

This guide provides instructions for building, saving, loading, and running the Docker container required to reproduce the `adaptive-mfd` results.

## Prerequisites

Ensure Docker is installed and running on your system. You can verify the installation by opening a terminal and running:

```bash
docker --version
```

If Docker is installed correctly, the command will display the installed Docker version.

## Build the Container

Navigate to the Docker directory:

```bash
cd adaptive-mfd/docker
```

Build the container from scratch:

```bash
docker build --no-cache --build-arg OS_TYPE=aarch64 -t adaptive_mfd:v1 .
```

## Save the Container Image

This step is only required when creating a Zenodo record containing the container image.

Save and compress the Docker image:

```bash
docker save adaptive_mfd:v1 | gzip > adaptive_mfd_repro.tar.gz
```

## Load the Container Image

To load the saved container image:

```bash
docker load -i adaptive_mfd_repro.tar.gz
```

## Run the Container

Start an interactive container session:

```bash
docker run -it --rm \
    --memory=16g \
    --memory-swap=16g \
    -v "$(pwd)":/app/workspace \
    -e PYTHONDONTWRITEBYTECODE=true \
    -e MPLLOCALFREETYPE=1 \
    adaptive_mfd:v1 \
    /bin/bash
```

## Run the Reproduction Script

Once inside the container, execute:

```bash
python main.py input/twoFault_liso.yaml
```

This command runs the example configuration and reproduces the corresponding `adaptive-mfd` results.
