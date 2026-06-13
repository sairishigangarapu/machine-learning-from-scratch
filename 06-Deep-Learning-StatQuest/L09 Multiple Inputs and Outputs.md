## Multiple Inputs and Outputs

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. Neural Networks with Multiple Inputs and Outputs

### Motivation and Intuition

So far we have seen neural networks with a single input and a single output. But real-world problems often have many input features and many output classes. For example, predicting iris species (setosa, versicolor, virginica) from both petal width and sepal width requires a network with 2 input nodes and 3 output nodes. Each input connects to each output through its own weight, and each output node has its own bias term. The entire forward pass can be expressed compactly using matrix multiplication.

### Matrix Form of Forward Propagation

Let $\mathbf{x} \in \mathbb{R}^n$ be the input vector, $\mathbf{W} \in \mathbb{R}^{m \times n}$ the weight matrix, and $\mathbf{b} \in \mathbb{R}^m$ the bias vector. The raw output (logits) $\mathbf{z} \in \mathbb{R}^m$ is:

$$
\mathbf{z} = \mathbf{W} \mathbf{x} + \mathbf{b}
$$

In expanded form for 2 inputs and 3 outputs:

$$
\begin{bmatrix} z_1 \\ z_2 \\ z_3 \end{bmatrix} =
\begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \\ w_{31} & w_{32} \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} +
\begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}
$$

Each output $z_j$ is:

$$
z_j = \sum_{i=1}^{n} w_{ji} x_i + b_j
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $x_i$ | Input feature $i$ | Raw data fed into the network |
| $w_{ji}$ | Weight connecting input $i$ to output $j$ | Controls how much input $i$ influences output $j$ |
| $b_j$ | Bias for output node $j$ | Allows the activation to shift regardless of input |
| $z_j$ | Raw output (logit) for class $j$ | Fed into activation/softmax for final prediction |
| $\mathbf{W}$ | Weight matrix, size $m \times n$ | Compact representation of all weight connections |
| $n$ | Number of input features | Dimensionality of the input data |
| $m$ | Number of output classes | Number of categories to predict |

Each raw output $z_j$ is then passed through an activation function (e.g., sigmoid for binary, softmax for multi-class) to produce the final prediction.

---

> **Check your intuition:** If you have 5 input features and 4 output classes, what are the dimensions of $\mathbf{W}$ and $\mathbf{b}$?

<details>
<summary>Answer</summary>
$\mathbf{W}$ is $4 \times 5$, $\mathbf{b}$ is $4 \times 1$.
</details>

---

## Prerequisites and Further Reading

- **Prerequisites:** L01 Neural Networks Part 0, L02 Neural Networks Part 1, basic matrix multiplication.
- **Next:** L10 ArgMax and SoftMax — converting raw outputs to probabilities.
