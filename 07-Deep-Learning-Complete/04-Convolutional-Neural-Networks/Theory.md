# Convolutional Neural Networks (CNNs)

## 1. Why CNNs?

Fully connected networks treat each pixel as an independent feature, destroying the **spatial structure** of images. For a 224x224 RGB image, a fully connected layer with 1000 outputs would have ~150 million parameters — computationally wasteful and prone to overfitting.

**CNNs** exploit the spatial hierarchy of images using three key ideas:

- **Local connectivity** — each neuron connects to a small local region (receptive field).
- **Parameter sharing** — the same filter (kernel) is reused across the entire spatial extent of the input.
- **Translation invariance** — a feature detector learned at one position works at any position; a cat is detected regardless of where it appears in the image.

This reflects a fundamental property of natural images: nearby pixels are strongly correlated, meaningful features repeat across space, and the hierarchical composition of edges to shapes to objects mirrors biological vision.

### Motivation and Intuition

Regular neural networks treat every input pixel independently via fully connected layers. For images, this ignores spatial structure — nearby pixels are strongly related, edges matter, and patterns repeat across the image. Convolutional Neural Networks use filters (kernels) that slide across the image, detecting local patterns like edges, corners, and textures. By stacking layers, CNNs build hierarchical representations: edges -> shapes -> object parts -> whole objects.

A convolution is simply a **sliding window multiplication and sum**. Take a small matrix (the kernel), slide it across the input image, and at each position compute the element-wise product and sum to produce a single output value. The result is a **feature map** showing where the kernel's pattern appears in the image.

| Property | Fully Connected | Convolutional |
| :--- | :--- | :--- |
| Connectivity | Every input to every output | Local receptive field |
| Parameters per layer | Massive (input_dim x output_dim) | Small (kernel_height x kernel_width x input_channels x output_channels) |
| Spatial structure | Ignored | Exploited |
| Translation equivariance | No | Yes (by construction) |
| Invariance to position | None | Strong (via pooling + weight sharing) |

---

## 2. Convolution Operation

### Motivation and Intuition

The convolution operation extracts local patterns from the input. A filter (kernel) slides across the input, computing a dot product at each position. This produces a **feature map** that highlights where the pattern detected by the filter appears.

The key insight is that the same filter is applied everywhere via **weight sharing**, dramatically reducing the number of parameters while making the detection translation-equivariant (if the feature moves, the activation moves correspondingly).

### Formal Definition

A filter (kernel) K in R^(h x w) slides over the input I in R^(H x W). At each position (i, j), the output (feature map) entry is:

$$
(\mathbf{I} * \mathbf{K})_{ij} = \sum_{u=1}^{h} \sum_{v=1}^{w} \mathbf{I}_{i+u, j+v} \cdot \mathbf{K}_{u, v}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| (I * K)_{ij} | Output feature map value at position (i,j) | The result of applying the filter at this position — one number in the output activation map |
| I_{i+u, j+v} | Input pixel value at position (i+u, j+v) | The local patch of the image being processed — defines the receptive field |
| K_{u, v} | Filter/kernel weight at position (u,v) | Learned parameters that detect specific features (edges, textures, etc.) |
| h, w | Height and width of the filter | Spatial extent of the local pattern being detected (typically 3x3 or 5x5) |
| H, W | Height and width of the input | Spatial dimensions of the full input image |

### Worked Numerical Example

Consider a 4x4 input and a 2x2 filter:

Input:
```
[1, 2, 3, 4]
[5, 6, 7, 8]
[9, 1, 2, 3]
[4, 5, 6, 7]
```

Filter (K):
```
[1, 0]
[0, -1]
```

Convolution output at (0,0) = 1*1 + 2*0 + 5*0 + 6*(-1) = 1 + 0 + 0 - 6 = -5

The full output (stride=1, no padding) is 3x3:
```
[-5, -5, -5]
[-4,  6,  6]
[ 5, -3, -5]
```

---

### Architecture Components

#### Filters/Kernels

Small learnable weight matrices (e.g., 3x3 or 5x5). Each filter detects one type of feature. Multiple filters produce multiple feature maps stacked along the channel dimension.

| Parameter | Description | Typical Value |
| :--- | :--- | :--- |
| Kernel size | Spatial dimensions of the filter | 3x3 (dominant in modern architectures) |
| Number of filters | Depth of the output feature map | Doubles after each pooling (32->64->128->256) |
| Initialization | How weights are initialized before training | He initialization (for ReLU) or Xavier/Glorot |

#### Stride

Step size when sliding the filter across the input. Stride > 1 downsamples the output.

Output spatial size given input size W, filter size F, stride S, and padding P:

$$
\text{Output\_Size} = \left\lfloor \frac{W - F + 2P}{S} \right\rfloor + 1
$$

| Stride | Effect | Use Case |
| :--- | :--- | :--- |
| 1 | Full resolution output | Standard convolution preserving spatial size (with padding) |
| 2 | Halves spatial dimensions | Built-in downsampling, replaces some pooling layers |

#### Padding

Adding zeros around the input border to control output spatial dimensions.

| Type | Description | Output Size (F=3, S=1) |
| :--- | :--- | :--- |
| Valid (no padding) | No zeros added, output shrinks | W - 2 |
| Same | Pad so output equals input size | W |
| Full | Pad so every input pixel is convolved equally | W + 2 |

**Same padding:** P = (F - 1) / 2 ensures the output has the same spatial dimensions as the input when stride = 1.

#### Feature Maps

The stacked outputs of all filters form the next layer's input. Early layers detect low-level features (edges, colors); later layers detect high-level features (faces, objects). This hierarchical feature learning is what makes CNNs so powerful.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Kernel K | Learnable weight filter | Detects local patterns in the input |
| Feature map | Output of one filter convolved over input | Activation map showing where features fire |
| Stride | Step size of filter movement | Controls spatial downsampling |
| Padding | Zero border added to input | Preserves spatial dimensions, controls output size |
| Receptive field | Region of input affecting one output neuron | Grows with depth; later layers see larger context |

---

## 3. Activation (ReLU)

### Motivation and Intuition

Convolution is a linear operation. Stacking linear operations without non-linearity would collapse to a single linear transformation, making deep networks pointless. Non-linear activation functions introduce the expressivity needed to learn complex patterns.

ReLU (Rectified Linear Unit) is the default choice because it is simple, fast, and helps mitigate the vanishing gradient problem.

### Formal Definition

$$
f(x) = \max(0, x)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| x | Input activation value | The output of the convolution or previous layer — can be any real number |
| f(x) | Output after ReLU | If x > 0, output is x; if x <= 0, output is 0 |
| max(0, x) | Element-wise maximum with zero | Kills all negative activations — creates sparse representations and solves the vanishing gradient problem for positive values |

### Why ReLU?

- **Non-linear** despite its piecewise-linear form.
- **Sparse activation** — negative neurons are completely shut off.
- **Vanishing gradient resistant** — gradient is 1 for positive inputs, 0 for negative, never small.
- **Computationally cheap** — just a max operation.

Applied element-wise after convolution (Conv -> ReLU is the standard pattern).

---

## 4. Pooling Layer

### Motivation and Intuition

After convolution, feature maps are large (same spatial size as the input). Pooling downsamples them, reducing computation in later layers and introducing **translation invariance** — small shifts in the input produce the same pooled output.

### Types

| Type | Operation | Effect |
| :--- | :--- | :--- |
| **Max Pooling** | Takes the maximum value in each window | Preserves the strongest features, ignores weak activations |
| **Average Pooling** | Takes the average in each window | Smoother representation, retains information about all activations |

### Formal Definition

Given a pooling window of size (p, p) and stride s (typically p = s = 2):

$$
\begin{aligned}
\text{MaxPool: } y_{ij} &= \max_{u=0}^{p-1} \max_{v=0}^{p-1} x_{i \cdot s + u, j \cdot s + v} \\
\text{AvgPool: } y_{ij} &= \frac{1}{p^2} \sum_{u=0}^{p-1} \sum_{v=0}^{p-1} x_{i \cdot s + u, j \cdot s + v}
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| y_{ij} | Output at position (i,j) | Downsampled value after pooling |
| x_{i*s+u, j*s+v} | Input pixel in the pooling window | The local region being aggregated |
| p | Pooling window size | Spatial extent of the aggregation (typically 2x2) |
| s | Stride of pooling | Step between windows (typically 2, same as p) |

Typical: 2x2 window with stride 2 halves spatial dimensions. A 28x28 feature map becomes 14x14, then 7x7 after two pooling layers.

### Worked Numerical Example

Input (4x4):
```
[1, 3, 2, 4]
[5, 6, 7, 8]
[9, 1, 0, 2]
[3, 4, 5, 6]
```

Max Pooling (2x2, stride 2):
```
max(1,3,5,6)=6  max(2,4,7,8)=8
max(9,1,3,4)=9  max(0,2,5,6)=6
```
Output:
```
[6, 8]
[9, 6]
```

Average Pooling (2x2, stride 2):
```
avg(1,3,5,6)=3.75  avg(2,4,7,8)=5.25
avg(9,1,3,4)=4.25  avg(0,2,5,6)=3.25
```
Output:
```
[3.75, 5.25]
[4.25, 3.25]
```

---

## 5. Fully Connected Layer

### Motivation and Intuition

After the convolutional base has extracted hierarchical spatial features, a standard classifier is needed to map these features to class probabilities. The feature maps are flattened into a 1D vector and passed through one or more fully connected layers.

This separation of concerns is fundamental: the convolutional layers learn **what** is in the image and **where** it is, while the fully connected layers learn **which combination of features** corresponds to each class.

### Architecture Pattern

After convolution and pooling, the feature maps are **flattened** and fed into a standard dense layer for classification:

```
Input Image
 |
[Conv -> ReLU -> Pool] x N   <- Feature extraction (convolutional base)
 |
[Flatten]                     <- Reshape for dense layers
 |
[Fully Connected -> ReLU] x M <- Classification head
 |
[Softmax Output]              <- Class probabilities
```

### CNN Forward Pass Summary

$$
\begin{aligned}
\mathbf{Z}^{(l)} &= \text{Conv}(\mathbf{A}^{(l-1)}, \mathbf{W}^{(l)}) + \mathbf{b}^{(l)} \\
\mathbf{A}^{(l)} &= \text{ReLU}(\mathbf{Z}^{(l)}) \\
\mathbf{A}^{(l+1)} &= \text{Pool}(\mathbf{A}^{(l)})
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Z^{(l)} | Pre-activation at layer l | Convolution output before non-linearity |
| A^{(l)} | Activation at layer l | Feature map after ReLU |
| W^{(l)} | Convolutional kernel at layer l | Learned filter weights |
| b^{(l)} | Bias at layer l | Shift parameter |

---

## 6. Classic CNN Architectures

### Motivation and Intuition

Since LeNet-5 in 1998, CNN architectures have evolved dramatically. Each major breakthrough introduced innovations that improved depth, efficiency, or accuracy. Understanding these architectures reveals the design principles that make deep learning work for vision.

### Architecture Timeline

| Year | Architecture | Key Innovation | Depth | Top-5 Error (ImageNet) |
| :--- | :--- | :--- | :--- | :--- |
| 1998 | **LeNet-5** | First practical CNN for digit recognition | 5 | N/A (MNIST) |
| 2012 | **AlexNet** | ReLU activation, Dropout, GPU training, data augmentation | 8 | 15.3% |
| 2014 | **VGGNet** | Uniform 3x3 filters, deep and simple design | 16-19 | 7.3% |
| 2014 | **GoogLeNet** | Inception modules (multi-scale convolutions), 1x1 bottleneck | 22 | 6.7% |
| 2015 | **ResNet** | Skip connections (residual learning), batch normalization | 50-152 | 3.6% |

### Key Contributions

**LeNet-5:** Established the Conv->Pool->Conv->Pool->FC pattern still used today. Applied to handwritten digit recognition (MNIST).

**AlexNet:** Demonstrated that deep CNNs win on large-scale image classification. Introduced ReLU (faster training than tanh), Dropout (regularization), and GPU parallelism.

**VGGNet:** Proved that stacking many 3x3 convolutions is better than using larger filters. Two 3x3 layers have the same receptive field as one 5x5 but with fewer parameters and more non-linearity.

**GoogLeNet (Inception):** Introduced Inception modules that apply 1x1, 3x3, and 5x5 convolutions in parallel, letting the network learn multi-scale features. Used 1x1 convolutions as bottleneck layers to reduce channel depth.

---

## 7. ResNet: Skip Connections

### Motivation and Intuition

Deep networks suffer from **degradation** — as depth increases, accuracy saturates and then drops, even with proper initialization and batch normalization. This is NOT caused by overfitting (training error also increases). Instead, it reflects the difficulty of learning identity mappings in very deep networks.

**Residual learning** (He et al., 2015) reformulates the problem. Instead of learning the desired mapping H(x) directly, the network learns the residual F(x) = H(x) - x, so the actual mapping is F(x) + x. If the identity mapping is optimal, the network can simply push F(x) toward zero — much easier than learning identity from scratch.

### Formal Definition

$$
\text{Output} = F(x) + x
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| x | Input to the residual block | The original feature map passed through the shortcut connection (identity skip) |
| F(x) | Residual function learned by the block | The output of the stacked convolutional layers — what the block learns to ADD to the input |
| F(x) + x | Skip connection output (element-wise addition) | The input plus the residual — if F(x) ~= 0, the block acts as identity, making deep networks easy to train |
| Shortcut connection | The direct path from input to output (bypass) | Allows gradients to flow directly backward through the network without vanishing |

### Why Skip Connections Work

1. **Gradient flow:** The gradient of the loss with respect to x flows directly through the addition operation (identity branch), avoiding repeated multiplication by small weights.
2. **Easy identity learning:** If a layer is not needed, the network learns to set its weights near zero, and the shortcut carries the input forward.
3. **Ensemble effect:** ResNets behave like ensembles of shallower networks, each path through the network containing a different subset of residual blocks.

### ResNet Block Variants

| Variant | Structure | Depth |
| :--- | :--- | :--- |
| Basic Block | 3x3 Conv -> BN -> ReLU -> 3x3 Conv -> BN | ResNet-18, 34 |
| Bottleneck Block | 1x1 -> 3x3 -> 1x1 (reduces then expands channels) | ResNet-50, 101, 152 |

This allows gradients to flow directly through the identity shortcut, enabling training of 100+ layer networks (ResNet-152 has 152 layers yet trains stably).

---

## 8. Data Augmentation

### Motivation and Intuition

CNNs have many parameters and can easily overfit to training data, especially with limited datasets. Data augmentation artificially expands the training set by applying random (but realistic) transformations to input images.

Each transformation creates a slightly different version of the same image, teaching the model to be invariant to these transformations. This is effectively a form of regularization.

### Common Techniques

| Technique | Description | Effect |
| :--- | :--- | :--- |
| **Random Cropping** | Extract random patches (e.g., 224x224 from 256x256) | Invariance to position and scale |
| **Horizontal Flip** | Mirror images left-right with 50% probability | Invariance to left-right orientation |
| **Rotation** | Rotate by random angle (e.g., +/-15 degrees) | Invariance to slight rotation |
| **Color Jitter** | Vary brightness, contrast, saturation, hue randomly | Invariance to lighting conditions |
| **Random Erasing** | Black out random rectangular regions | Robustness to occlusion |
| **Mixup** | Linearly blend pairs of images and labels | Softens decision boundaries, regularizes |
| **CutMix** | Cut a patch from one image and paste onto another | Combines cropping and mixing |

Modern practice uses **automatic augmentation** (AutoAugment, RandAugment) — learned augmentation policies discovered by searching over augmentation spaces.

> **Check your intuition:** A 32x32 grayscale image is passed through a 3x3 filter with stride 1 and padding 1. What is the output size?

> **Answer:** 32x32 (same padding preserves spatial dimensions: P = (F-1)/2 = 1, so Output = (32 - 3 + 2*1)/1 + 1 = 32).

---

## 9. Code Example: CNN on MNIST

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),    # 1x28x28 -> 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 32x28x28 -> 32x14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 32x14x14 -> 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 64x14x14 -> 64x7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)                    # 10 digits
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Alternative style using functional API
class SimpleCNNFunctional(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)    # 7x7 after 2x pooling on 28x28

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))    # [B, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))    # [B, 32, 7, 7]
        x = x.view(x.size(0), -1)              # flatten
        x = self.fc(x)                         # [B, num_classes]
        return x

model = SimpleCNN()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Forward pass with random data
x = torch.randn(32, 1, 28, 28)    # batch=32, 1 channel, 28x28
out = model(x)
print(f"Output shape: {out.shape}")  # (32, 10)
```

---

## 10. Key Hyperparameters

| Hyperparameter | Typical Values | Effect |
| :--- | :--- | :--- |
| Learning rate | 1e-3 (Adam), 1e-2 (SGD + momentum) | Controls step size during gradient descent |
| Batch size | 32, 64, 128 | Trade-off: larger = stable gradients but more memory |
| Kernel size | 3x3 (dominant in modern architectures) | Larger = bigger receptive field but more parameters |
| Number of filters | 32 -> 64 -> 128 -> 256 (double after pooling) | Controls model capacity |
| Dropout | 0.25-0.5 in fully connected layers | Regularization — randomly drops neurons during training |
| Weight decay | 1e-4 to 5e-4 | L2 regularization penalty |

---

## 11. Advantages and Disadvantages

### Pros

- Exploits spatial structure — far fewer parameters than fully connected (parameter sharing).
- Translation invariant by construction (via pooling and weight sharing).
- State-of-the-art for image tasks (classification, detection, segmentation).
- Hierarchical feature learning (edges -> textures -> shapes -> objects).
- Transfer learning works well — pre-trained CNNs adapt to new tasks with little data.

### Cons

- Computationally expensive (GPU recommended for training).
- Requires large datasets (mitigated by transfer learning and data augmentation).
- Not naturally suited for sequential or tabular data.
- Black box — interpretability requires tools like Grad-CAM or feature visualization.
- Sensitive to adversarial examples — small perturbations can fool the network.
- Less effective when spatial structure is weak (e.g., some medical imaging modalities).

---

## Prerequisites and Further Reading

- **Prerequisites:** Artificial Neural Networks, Backpropagation, Softmax and Cross-Entropy Loss.
- **Related:** RNN (sequential data counterpart), Transformers (alternative for vision — ViT).
- **Original papers:**
  - LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (LeNet, 1998)
  - Krizhevsky et al., "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet, 2012)
  - Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition" (VGG, 2014)
  - Szegedy et al., "Going Deeper with Convolutions" (GoogLeNet, 2014)
  - He et al., "Deep Residual Learning for Image Recognition" (ResNet, 2015)
- **Next:** Recurrent Neural Networks (RNNs) for sequential data.
