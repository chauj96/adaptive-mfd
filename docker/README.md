# Docker — Reproducible Adaptive MFD Solver

## Build

```bash
docker build -t adaptive-mfd docker/
```

### Build options

| ARG          | Default  | Description                     |
|--------------|----------|---------------------------------|
| `UBUNTU_VER` | `latest` | Ubuntu base image tag           |
| `PY_VER`     | `3.12`   | Python version                  |
| `OS_TYPE`    | `x86_64` | Miniforge arch (`x86_64`/`aarch64`) |

Example with a specific Python version:

```bash
docker build --build-arg PY_VER=3.11 -t adaptive-mfd docker/
```

## Run

Run the default test case (fully polyhedral):

```bash
docker run --rm adaptive-mfd
```

Run a specific input file:

```bash
docker run --rm adaptive-mfd python main.py input/twoFault_liso.yaml
docker run --rm adaptive-mfd python main.py input/twoFault_hetani.yaml
docker run --rm adaptive-mfd python main.py input/spe11b.yaml
```

Save output files to the host:

```bash
docker run --rm -v $(pwd)/output:/app/adaptive-mfd/output adaptive-mfd
```
