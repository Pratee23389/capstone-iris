# Iris Logistic Regression (GPU)

## What
Train a logistic‐regression model on the Iris dataset using PyTorch, fully on GPU.

## Requirements
- Python 3.7+  
- NVIDIA GPU + CUDA drivers  
- Install dependencies:
  ```bash
  pip install -r requirements.txt
# Iris Logistic Regression on GPU

[![PyTorch](https://img.shields.io/badge/pytorch-2.0.0-blue.svg)](https://pytorch.org/)  
[![Python](https://img.shields.io/badge/python-3.7%2B-yellow.svg)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A minimal end‑to‑end example of training a **logistic regression** model on the classic Iris dataset **fully on GPU** using PyTorch. Demonstrates the complete workflow: data loading & preprocessing, tensor transfers to CUDA device, model definition & training, validation, and model export.

---

## 📋 Table of Contents

- [Demo](#-demo)  
- [Prerequisites](#-prerequisites)  
- [Installation](#-installation)  
- [Usage](#-usage)  
  - [Command‑Line Arguments](#command‑line‑arguments)  
- [Project Structure](#-project-structure)  
- [Results & Proof](#-results--proof)  
- [Next Steps](#-next-steps)  
- [Contributing](#-contributing)  
- [License](#-license)  

---

## 👀 Demo

```bash
$ ./iris_logreg.py --epochs 100 --lr 0.1 --test_size 0.2
Using device: cuda
Epoch [ 20/100] Loss: 0.7213  Val Acc: 80.00%
Epoch [ 40/100] Loss: 0.5412  Val Acc: 86.67%
Epoch [ 60/100] Loss: 0.3397  Val Acc: 90.00%
Epoch [ 80/100] Loss: 0.2375  Val Acc: 93.33%
Epoch [100/100] Loss: 0.1543  Val Acc: 93.33%
Model saved to model.pt
