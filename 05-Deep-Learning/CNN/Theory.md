# Convolutional Neural Networks (CNNs) 🖼️

## 1. Why CNNs?

Fully connected networks treat each pixel as an independent feature, destroying the **spatial structure** of images. For a 224×224 RGB image, a fully connected layer with 1000 outputs would have ~150 million parameters — computationally wasteful and prone to overfitting.

**CNNs** exploit the spatial hierarchy of images using three key ideas:
* **Local connectivity** — each neuron connects to a small region.
* **Parameter sharing** — the same filter is reused across the entire image.
* **Translation invariance** — a cat is detected regardless of its position.

---

## 2. Core Building Blocks

### A. Convolution Layer
Applies a set of **filters** (kernels) to the input. Each filter slides across the image computing element-wise multiplications and sums:

$$
\text{Output}(i,j) = \sum_{m}\sum_{n} \text{Input}(i+m, j+n) \cdot \text{Filter}(m,n) + b
$$

| Parameter | Description |
| :--- | :--- |
| **Kernel size** | Spatial dimensions of the filter (e.g., 3×3, 5×5) |
| **Stride** | Step size when sliding the filter |
| **Padding** | Zero-padding around the border (preserves spatial dimensions) |
| **Number of filters** | Determines the depth of the output (e.g., 32, 64) |

### B. Activation (ReLU)
$$
f(x) = \max(0, x)
$$
Introduces non-linearity. Applied element-wise after convolution.

### C. Pooling Layer
Reduces spatial dimensions while retaining important features:

| Type | Operation | Effect |
| :--- | :--- | :--- |
| **Max Pooling** | Takes the maximum value in each window | Preserves the strongest features |
| **Average Pooling** | Takes the average in each window | Smoother representation |

Typical: 2×2 window with stride 2 → halves spatial dimensions.

### D. Fully Connected Layer
After convolution and pooling, the feature maps are **flattened** and fed into a standard dense layer for classification.

---

## 3. CNN Architecture Pattern

```
Input Image
    ↓
[Conv → ReLU → Pool] × N    ← Feature extraction (convolutional base)
    ↓
[Flatten]                    ← Reshape for dense layers
    ↓
[Fully Connected → ReLU] × M ← Classification head
    ↓
[Softmax Output]             ← Class probabilities
```

### Classic Architectures

| Year | Architecture | Key Innovation | Depth |
| :--- | :--- | :--- | :--- |
| 1998 | **LeNet-5** | First practical CNN | 5 |
| 2012 | **AlexNet** | ReLU, Dropout, GPU training | 8 |
| 2014 | **VGGNet** | Uniform 3×3 filters | 16-19 |
| 2014 | **GoogLeNet** | Inception modules (multi-scale) | 22 |
| 2015 | **ResNet** | Skip connections (residual learning) | 50-152 |

---

## 4. ResNet: Skip Connections

Deep networks suffer from **degradation** — accuracy saturates and then drops. **Residual learning** addresses this by learning $F(x) = H(x) - x$ instead of $H(x)$ directly:

$$
\text{Output} = F(x) + x
$$

This allows gradients to flow directly through the identity shortcut, enabling training of 100+ layer networks.

---

## 5. Code Example: CNN on MNIST

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 1→32 channels
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 28x28 → 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32→64 channels
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 14x14 → 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 digits
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 6. Data Augmentation

To prevent overfitting, CNNs benefit from artificially expanding the training set:

| Technique | Example |
| :--- | :--- |
| **Random Cropping** | Extract random 224×224 patches |
| **Horizontal Flip** | Mirror images randomly |
| **Rotation** | Rotate ±15° |
| **Color Jitter** | Vary brightness, contrast, saturation |

---

## 7. Key Hyperparameters

| Hyperparameter | Typical Values |
| :--- | :--- |
| Learning rate | 1e-3 (Adam), 1e-2 (SGD + momentum) |
| Batch size | 32, 64, 128 |
| Kernel size | 3×3 (dominant in modern architectures) |
| Number of filters | Double after each pooling (32→64→128→256) |
| Dropout | 0.25–0.5 in fully connected layers |

---

## 8. Advantages & Disadvantages

### ✅ Pros
* Exploits spatial structure — far fewer parameters than fully connected.
* Translation invariant by construction.
* State-of-the-art for image tasks.
* Hierarchical feature learning (edges → textures → objects).

### ❌ Cons
* Computationally expensive (GPU recommended).
* Requires large datasets (mitigated by transfer learning).
* Not naturally suited for sequential or tabular data.
* Black box — interpretability requires tools like Grad-CAM.

---

**Previous:** [RNN](../RNN/Theory.md) | **Related:** [ANN](../../02-Supervised-Learning/ARTIFICIAL%20NEURAL%20NETWORKS/Theory.md) | **Related:** [PCA](../../03-Unsupervised-Learning/PRINCIPAL%20COMPONENT%20ANALYSIS/Theory.md)
