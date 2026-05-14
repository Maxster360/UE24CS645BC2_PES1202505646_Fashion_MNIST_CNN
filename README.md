# Fashion MNIST CNN from Scratch

**Course:** UE24CS645BC2 - Deep Learning: Theory and Practice  
**USN:** PES1202505646  
**Assignment:** DLTP Assignment 1

## About

A Convolutional Neural Network (CNN) built entirely from scratch using only NumPy — no deep learning frameworks (PyTorch, TensorFlow, etc.) are used. The implementation demonstrates the fundamental operations of CNNs including convolution, pooling, flattening, and fully connected layers with complete forward and backward propagation.

## Architecture

```
Input (28x28 grayscale image)
    │
    ▼
Conv Layer 1 (8 filters, 3x3) → ReLU → MaxPool (2x2)
    │
    ▼
Conv Layer 2 (16 filters, 3x3) → ReLU → MaxPool (2x2)
    │
    ▼
Flatten (16 × 5 × 5 = 400)
    │
    ▼
Fully Connected Layer (400 → 128) → ReLU
    │
    ▼
Fully Connected Layer (128 → 10) → Softmax
    │
    ▼
Output (10 classes)
```

## Components Implemented

| Component | Description |
|-----------|-------------|
| `ConvolutionLayer` | Convolution operation with configurable filters, stride, and padding |
| `MaxPoolLayer` | Max pooling with configurable pool size and stride |
| `ReLU` | Rectified Linear Unit activation |
| `FlattenLayer` | Reshapes multi-dimensional tensor to 1D vector |
| `FullyConnectedLayer` | Dense layer with weight matrix multiplication |
| `Softmax` | Softmax activation with cross-entropy loss |
| `CNN` | Complete model orchestrating forward and backward passes |

## Dataset

**Fashion MNIST** — 70,000 grayscale images (28x28) across 10 classes:

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

The dataset is automatically downloaded when running the training script.

## How to Run

### Prerequisites

- Python 3.8+
- NumPy

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/UE24CS645BC2_PES1202505646_Fashion_MNIST_CNN.git
cd UE24CS645BC2_PES1202505646_Fashion_MNIST_CNN

# Install dependencies
pip install -r requirements.txt
```

### Training and Evaluation

```bash
python train.py
```

This will:
1. Download the Fashion MNIST dataset (if not already present)
2. Train the CNN for 3 epochs on 1000 samples per epoch
3. Evaluate on 500 test samples
4. Print per-class accuracy

## File Structure

```
├── cnn.py          # CNN implementation (all layers, forward/backward pass)
├── train.py        # Training loop, evaluation, and data loading
├── requirements.txt
└── README.md
```

## Key Implementation Details

- **Weight Initialization:** He initialization (√(2/n)) for stable gradient flow
- **Loss Function:** Cross-entropy loss via softmax output
- **Optimization:** Stochastic Gradient Descent (SGD)
- **Backpropagation:** Full gradient computation through all layers including convolution and pooling

## Note

Since this is a pure NumPy implementation without GPU acceleration, training is computationally intensive. The default configuration uses a subset of the training data (1000 samples/epoch) to demonstrate functionality within a reasonable time frame. Increase `num_samples` in `train.py` for better accuracy at the cost of longer training time.
