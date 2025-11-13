
# Docker Deployment Guide

## Quick Start

### Prerequisites

- Docker Desktop installed and running

### Build and Run

```bash

# Build the image (first time only, takes 5-10 min)

docker-compose build

# Start the container

docker-compose up

# Access Jupyter at: http://localhost:8888

```

### Stop Container

```bash

# Press Ctrl+C in the terminal

# Or run:

docker-compose down

```

## What's Included

 Python 3.10 environment

 All dependencies (PyTorch, Transformers, scikit-learn, etc.)

 Jupyter Notebook server

 All project code and notebooks

 Volume mounts for live editing

## Troubleshooting

**Port 8888 already in use?**

Change port in docker-compose.yml:

```yaml

ports:

  - "8889:8888"

```

**Out of memory?**

Increase Docker memory:

Docker Desktop → Settings → Resources → Memory → 8GB

## Commands

```bash

# View logs

docker-compose logs -f

# Enter container shell

docker exec -it anli-classification bash

# Clean up

docker-compose down

docker system prune -a

```

## Notes

- Models are mounted from local directory (not in image)

- First run downloads dependencies

- Changes to code reflect immediately via volumes

