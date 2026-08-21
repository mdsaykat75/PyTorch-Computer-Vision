# FashionMNIST Image Classification with PyTorch

A deep learning project that classifies clothing images into 10 categories using a Convolutional Neural Network (TinyVGG) built from scratch in PyTorch. The dataset is loaded directly from raw binary files.

---

## Overview

This project implements the full computer vision pipeline:

1. Load raw FashionMNIST IDX files using NumPy and Python's `struct` module
2. Build a custom CNN (TinyVGG) using `torch.nn.Module`
3. Train the model with a complete training/validation loop
4. Evaluate test accuracy
5. Visualize training loss and sample predictions

---

## Dataset: FashionMNIST

FashionMNIST contains **70,000 grayscale clothing images** (28x28 pixels) across 10 categories.

| Label | Class |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

- **Training set:** 60,000 images
- **Test set:** 10,000 images
- **Format:** Raw binary IDX files (no torchvision dependency)
- **Source:** http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/

---

## Project Structure

```
pytorch-cv-project/
├── main.py                  # All code: data loading → model → training → visualization
├── requirements.txt         # Python dependencies
├── .gitignore              # Files excluded from Git
├── README.md               # This file
├── loss_curve.png          # Training loss plot (generated after running)
├── sample_predictions.png  # 16 sample predictions (generated after running)
└── data/
    └── fashionmnist/
        ├── train-images-idx3-ubyte
        ├── train-labels-idx1-ubyte
        ├── t10k-images-idx3-ubyte
        └── t10k-labels-idx1-ubyte
```

---

## Requirements

```
torch
numpy
matplotlib
```


---

## How to Run

### 1. Download the dataset

Download the 4 raw IDX files from the FashionMNIST website and place them in `data/fashionmnist/`:

```
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

Make sure the files have **no file extension** (not `.gz`). If they are still zipped, extract them first.

### 2. Run the project

```bash
main.py
```

The script will:
- Load and preprocess the data
- Print model architecture and device (CPU or CUDA)
- Train for 10 epochs, printing loss every 100 steps
- Evaluate test accuracy after each epoch
- Save `loss_curve.png` and `sample_predictions.png`

---



## Model Architecture: TinyVGG

```
Input: (1 x 28 x 28) grayscale image

Block 1:
  Conv2d(1 -> 10, kernel=3, padding=1) -> ReLU
  Conv2d(10 -> 10, kernel=3, padding=1) -> ReLU
  MaxPool2d(2)
  Output: (10 x 14 x 14)

Block 2:
  Conv2d(10 -> 10, kernel=3, padding=1) -> ReLU
  Conv2d(10 -> 10, kernel=3, padding=1) -> ReLU
  MaxPool2d(2)
  Output: (10 x 7 x 7)

Classifier:
  Flatten -> Linear(490 -> 10)
  Output: 10 class logits
```

| Design Choice | Reason |
|---------------|--------|
| 3x3 convolutions | Standard filter size, captures local patterns |
| Padding=1 | Preserves spatial dimensions after convolution |
| MaxPool2d(2) | Halves spatial size, adds translation invariance |
| ReLU | Introduces non-linearity |
| CrossEntropyLoss | Standard for multi-class classification |
| Adam optimizer | Adapts learning rate per parameter |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 10 |
| Device | CUDA if available, otherwise CPU |

---

## Results

| Metric | Value  |
|--------|--------|
| Test Accuracy | 90.48% |
| Final Training Loss | 0.21   |

### Training Loss Curve

![Loss Curve](loss_curve.png)

### Sample Predictions

![Sample Predictions](sample_predictions.png)

Green = correct prediction, Red = incorrect prediction.

---

## Key Features

- **Device agnostic** — automatically uses GPU (CUDA) if available, falls back to CPU
- **Modular design** — data loading, model, training, and evaluation are cleanly separated
- **Reproducible** — all hyperparameters documented and configurable at the top of `main.py`

---

## Future Improvements

- Add data augmentation (random rotation, horizontal flip)
- Increase model depth (more conv blocks, batch normalization)
- Experiment with different optimizers (SGD with momentum, RMSprop)
- Add learning rate scheduling
- Extend to a larger dataset (CIFAR-10, custom images)
- Deploy as a web app using Gradio or Streamlit

---

## Author

Ash
