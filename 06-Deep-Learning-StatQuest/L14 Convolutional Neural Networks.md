## Convolutional Neural Networks

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. Convolutional Neural Networks

### Motivation and Intuition

Regular neural networks treat every input pixel independently via fully connected layers. For images, this ignores spatial structure — nearby pixels are strongly related, edges matter, and patterns repeat across the image. Convolutional Neural Networks (CNNs) use filters (kernels) that slide across the image, detecting local patterns like edges, corners, and textures. By stacking layers, CNNs build hierarchical representations: edges $\to$ shapes $\to$ object parts $\to$ whole objects.

### Convolution Operation

A filter (kernel) $\mathbf{K} \in \mathbb{R}^{h \times w}$ slides over the input $\mathbf{I} \in \mathbb{R}^{H \times W}$. At each position $(i, j)$, the output (feature map) entry is:

$$
(\mathbf{I} * \mathbf{K})_{ij} = \sum_{u=1}^{h} \sum_{v=1}^{w} \mathbf{I}_{i+u, j+v} \cdot \mathbf{K}_{u, v}
$$

### Architecture Components

**Filters/Kernels:** Small learnable weight matrices (e.g., $3 \times 3$ or $5 \times 5$). Each filter detects one type of feature. Multiple filters produce multiple feature maps.

**Stride:** Step size for sliding the filter. Stride $> 1$ downsamples the output.

**Padding:** Adding zeros around the input border to control output size. "Same" padding preserves input size.

**Feature Maps:** The stacked outputs of all filters form the next layer's input. Early layers detect low-level features (edges); later layers detect high-level features (faces, objects).

**Pooling:** Reduces spatial dimensions and adds translation invariance.

- **Max Pooling:** Takes the maximum value in each window.
- **Average Pooling:** Takes the average value in each window.

### CNN Forward Pass Summary

$$
\mathbf{Z}^{(l)} = \text{Conv}(\mathbf{A}^{(l-1)}, \mathbf{W}^{(l)}) + \mathbf{b}^{(l)}
$$
$$
\mathbf{A}^{(l)} = \text{ReLU}(\mathbf{Z}^{(l)})
$$
$$
\mathbf{A}^{(l+1)} = \text{Pool}(\mathbf{A}^{(l)})
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Kernel $\mathbf{K}$ | Learnable weight filter | Detects local patterns in the input |
| Feature map | Output of one filter convolved over input | Activation map showing where features fire |
| Stride | Step size of filter movement | Controls spatial downsampling |
| Padding | Zero border added to input | Preserves spatial dimensions |
| Pooling | Downsampling operation | Reduces size, adds invariance |
| Receptive field | Region of input affecting one output neuron | Grows with depth; later layers see larger context |

### Why CNNs Work Well for Images

1. **Local connectivity:** Each neuron connects to only a small region (locality of pixels).
2. **Weight sharing:** The same filter is applied everywhere (translation equivariance — a cat is a cat no matter where it appears).
3. **Hierarchical features:** Low-level to high-level feature composition.
4. **Fewer parameters:** Compared to fully connected layers on the same image size.

### Python Code: Simple CNN with PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                               kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)  # 7x7 after 2x pooling on 28x28

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 7, 7]
        x = x.view(x.size(0), -1)             # flatten
        x = self.fc(x)                        # [B, num_classes]
        return x
```

---

> **Check your intuition:** A $32 \times 32$ grayscale image is passed through a $3 \times 3$ filter with stride 1 and padding 1. What is the output size?

<details>
<summary>Answer</summary>
$32 \times 32$ (same padding preserves spatial dimensions).
</details>

---

## Prerequisites and Further Reading

- **Prerequisites:** L02 Neural Networks Part 1, L08 ReLU in Action, L09 Multiple Inputs and Outputs.
- **Next:** L15 Recurrent Neural Networks.
