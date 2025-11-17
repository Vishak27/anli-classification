# ANLI Multi-Class Classification

Natural Language Inference using Adversarial NLI dataset.

## Quick Start with Docker
```bash
docker-compose build
docker-compose up
```

Access Jupyter at: http://localhost:8888

## Models Trained

- BERT-base: 42.9% dev accuracy
- RoBERTa-base: 44.2% dev accuracy  
- DeBERTa-v3-base: 45.3% dev accuracy 

See DOCKER.md for detailed deployment instructions.

## 🐳 Docker Deployment

### Quick Start
```bash
# Build and run
docker-compose up --build

# Access Jupyter at http://localhost:8888
```

### Features
✅ Reproducible environment
✅ All dependencies included
✅ One-command setup
✅ Volume mounts for live editing

See [DOCKER.md](DOCKER.md) for detailed instructions.
