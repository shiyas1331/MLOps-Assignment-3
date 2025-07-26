# MLOps Assignment 3

## Project Summary

This assignment builds an end-to-end MLOps pipeline involving:

- Model training using Scikit-learn & PyTorch
- Docker containerization
- CI/CD using GitHub Actions
- Manual 8-bit quantization
- Performance comparison between original and quantized models

---

## Dataset & Model

- **Dataset:** California Housing (`sklearn.datasets`)
- **Original Model:** Linear Regression (Scikit-learn)
- **Quantized Model:** Single-layer PyTorch model with 8-bit quantized weights

---

## Quantization Comparison

| Metric             | Original Sklearn | Quantized Model |
|--------------------|------------------|------------------|
| **R² Score**       | 0.605            | 0.605            |
| **Model Size (KB)**| 0.4 KB           | 0.4 KB           |

---

## 📂 Branches

- `main`: Initial setup
- `dev`: Model development
- `docker_ci`: Docker + CI/CD pipeline
- `quantization`: Manual quantization + analysis

---

## Links

- **GitHub Repo**: https://github.com/shiyas1331/MLOps-Assignment-3
