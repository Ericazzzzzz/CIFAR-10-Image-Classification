# CIFAR-10 Image Classification with Structured CNN Architecture

This project implements and evaluates two convolutional neural networks (CNNs) to classify images in the CIFAR-10 dataset, using a structured architecture composed of **intermediate blocks** and an **output block**. The models are modular, interpretable, and optimized for high performance.

## Table of Contents

- [📝 Overview](#-overview)
- [🖼️ Dataset](#-dataset)
- [🏗️ Architectural Framework](#-architectural-framework)
  - [Intermediate Blocks](#-intermediate-blocks)
  - [Output Block](#-output-block)
- [🧪 Implemented Models](#-implemented-models)
  - [CIFAR10Net (Baseline)](#️-cifar10net-baseline)
  - [Improved_CIFAR10Net](#-improved_cifar10net)
- [🛠️ Training Details](#-training-details)
- [📊 Results](#-results)
- [🔗 References](#-references)

---

## 📝 Overview

Two models were developed:

- **CIFAR10Net** – A simple baseline model using a structured CNN architecture.
- **Improved_CIFAR10Net** – A deeper model integrating additional convolutional layers, dropout, and data augmentation to reach higher accuracy.

Both models follow a strict block-based design that ensures interpretability and modularity.

---

## 🖼️ Dataset

The project uses the CIFAR-10 dataset:

- 60,000 images (32×32 RGB), 10 classes  
  - 50,000 for training  
  - 10,000 for testing
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Dataset source: https://www.cs.toronto.edu/~kriz/cifar.html

---

## 🏗️ Architectural Framework

### Intermediate Blocks

Each intermediate block receives input `x` and applies `L` independent convolutional layers in parallel. The outputs are combined via a **softmax-weighted sum**, where the weights are learned from the input.

Formula: x′ = a₁·C₁(x) + a₂·C₂(x) + ... + aL·CL(x)


How the weights `a` are computed:
- Average each channel of `x` → vector `m`
- Pass `m` through a fully connected layer
- Apply softmax → normalized weight vector `a`

Each block may also include:
- Batch Normalization
- ReLU Activation
- Max Pooling
- Dropout (optional)

> All outputs in a block must be the same shape to allow combination.

### Output Block

The output block transforms the final feature map into logits:

1. Average channel values → vector `m`
2. Pass through one or more fully connected layers (optional)
3. Output a 10-dimensional logits vector for classification

---

## 🧪 Implemented Models

### CIFAR10Net (Baseline)

- 3 intermediate blocks  
  - Each with 3 conv layers (32 → 64 channels)  
  - ReLU, BatchNorm, MaxPool  
- Output block with no hidden layers

**Training**:
- Optimizer: Adam (lr=0.01)  
- Epochs: 20  
- Batch Size: 128  
- Weight Init: Kaiming

**Test Accuracy**: 75%

---

### Improved_CIFAR10Net

- 5 intermediate blocks with increasing complexity
  - Block 1: 3 convs + extra conv + dropout (20%)
  - Block 2: 6 convs + maxpool + dropout (20%)
  - Block 3: 8 convs + extra conv + maxpool + dropout (40%)
  - Block 4: 6 convs + dropout (40%)
  - Block 5: 3 convs + maxpool + dropout (50%)
- Output block with 2 hidden layers: [256, 128]

**Training Techniques**:
- Data Augmentation: random crop, horizontal flip, normalization
- Scheduler: Cosine Annealing (T_max=200)
- Epochs: 100  
- Batch Size: 64

**Test Accuracy**: **92%**

---

## 🛠️ Training Details

- Framework: PyTorch  
- Loss: CrossEntropyLoss  
- Evaluation:
  - Loss per training batch
  - Accuracy per epoch (train/test)
- Visualizations: Loss and accuracy curves plotted

---

## 📊 Results

| Model               | Epochs | Test Accuracy |
|--------------------|--------|---------------|
| CIFAR10Net         | 20     | 75%           |
| Improved_CIFAR10Net| 100    | **92%**       |

---

## 🔗 References

- Gautam, A., Lohumi, Y., & Gangodkar, D. (2024). *Achieving Near-Perfect Accuracy in CIFAR-10 Classification*. [DOI](https://doi.org/10.1109/icait61638.2024.10690610)
- Kamal Das (2021). *CIFAR10 ResNet: 90+% accuracy in <5 min*. [Kaggle](https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min)
- Pandit, S. & Kumar, S. (2020). *Improved CNN for CIFAR-10 Image Classification*. [DOI](https://doi.org/10.5120/ijca2020920489)


