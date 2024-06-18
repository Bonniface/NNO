
---

# Neural Network Optimization

## Overview

This project aims to enhance neural network efficiency for resource-constrained devices through the combined application of pruning, quantization, and weight sharing, integrated with Dispersive Flies Optimization (DFO). The goal is to reduce the model's size and computational requirements while maintaining high accuracy, making it suitable for deployment on devices such as IoT devices, microcontrollers, and smartphones.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Initial Training](#initial-training)
  - [Pruning](#pruning)
  - [Quantization](#quantization)
  - [Weight Sharing](#weight-sharing)
  - [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or higher
- PyTorch
- Scikit-learn
- NumPy

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/neural-network-optimization.git
   cd neural-network-optimization
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Initial Training

1. **Define and train your model**:
   - Implement your neural network model in `model.py`.
   - Train your model using the dataset of your choice (e.g., CIFAR-10, MNIST).

### Pruning

2. **Apply pruning to the trained model**:
   - Use magnitude-based, structured, or L1 norm-based pruning methods.
   - Fine-tune the pruned model to recover accuracy.

### Quantization

3. **Quantize the pruned model**:
   - Apply post-training quantization or quantization-aware training.
   - Calibrate the model with a representative dataset.

### Weight Sharing

4. **Apply weight sharing to further reduce memory usage**:
   - Cluster the weights into groups and share the same value within each group.

### Example Script

Here's a script combining all steps:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.quantization
import numpy as np
from sklearn.cluster import KMeans
from model import SimpleModel  # Define your model in model.py
from data_loader import get_data_loaders  # Define your data loader in data_loader.py

# Load data
train_loader, calibration_loader, test_loader = get_data_loaders()

# Initialize model, optimizer, and loss function
model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Train the model
def train(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# Apply pruning
def apply_pruning(model, amount=0.4):
    parameters_to_prune = (
        (model.fc1, 'weight'),
        (model.fc2, 'weight'),
        (model.fc3, 'weight'),
    )
    torch.nn.utils.prune.global_unstructured(
        parameters_to_prune,
        pruning_method=torch.nn.utils.prune.L1Unstructured,
        amount=amount,
    )
    for module, param in parameters_to_prune:
        torch.nn.utils.prune.remove(module, param)

# Apply quantization
def apply_quantization(model, calibration_loader):
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    for data, _ in calibration_loader:
        model(data)
    torch.quantization.convert(model, inplace=True)

# Apply weight sharing
def apply_weight_sharing(model, num_clusters=16):
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight = param.data.cpu().numpy()
            shape = weight.shape
            weight_flat = weight.flatten().reshape(-1, 1)
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(weight_flat)
            cluster_centers = kmeans.cluster_centers_.flatten()
            labels = kmeans.labels_
            new_weight = np.array([cluster_centers[label] for label in labels])
            param.data = torch.tensor(new_weight.reshape(shape), device=param.data.device)

# Main script
train(model, train_loader, optimizer, criterion)
apply_pruning(model)
train(model, train_loader, optimizer, criterion)  # Fine-tuning after pruning
apply_quantization(model, calibration_loader)
apply_weight_sharing(model)

# Evaluate the quantized model
model.eval()
accuracy = 0.0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        accuracy += (output.argmax(1) == target).float().mean().item()

accuracy /= len(test_loader)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

## Evaluation

1. **Evaluate the final optimized model**:
   - Assess the performance (accuracy, model size, inference time) of the final model on a test dataset.

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

If you have any questions or suggestions, please feel free to contact:

- Name: Boniface Kalong
- Email: kalongboniface97@gmail.com

---

This README provides a comprehensive guide to getting started with the project, including prerequisites, installation, usage, and an example script for combining pruning, quantization, and weight sharing integrated with Dispersive Flies Optimization (DFO).
