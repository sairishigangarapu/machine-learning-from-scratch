## PCA: Mathematical Derivation and Examples

*Essential Mathematics for ML — Advanced Structured Notes*

---

## 1. The Core Objective of PCA

### Motivation and Intuition
In the previous modules, we established that Principal Component Analysis (PCA) relies on eigenvectors. But *why*? Why do eigenvectors magically preserve the structure of the data? 

The objective of PCA is essentially an optimization problem: **perform dimensionality reduction from a high-dimensional space ($n$) to a lower-dimensional subspace ($m$), while preserving as much of the original variance (randomness) as mathematically possible.**

To achieve this, we must strictly minimize the mathematical **Representation Error**. 

---

## 2. Rigorous Mathematical Derivation

Before we start doing calculus, let's explicitly define the players in this equation:

* **$\mathbf{x}$ (The Data):** A single raw data vector (e.g., an image with $n$ pixels).
* **$\phi_i$ (The Basis Direction):** A geometric axis in space. Think of the X-axis, Y-axis, etc.
* **$y_i$ (The Coordinate):** The specific actual value of our data $\mathbf{x}$ along the axis $\phi_i$. 
* **$b_i$ (The Constant):** A static "best guess" number we will use to replace $y_i$ when we compress the data.

### Step 1: Defining the Representation (The Ultimate Grid)
Let $\mathbf{x}$ be an $n$-dimensional data vector. To describe this perfectly without losing any detail, we can build it using a linear combination of a full, "perfect" grid of basis directions $\{\phi_1, \phi_2, \dots, \phi_n\}$. Assume these directions are orthonormal (at perfect 90-degree angles to each other).

$$
\mathbf{x} = \sum_{i=1}^{n} y_i \phi_i
$$

* **What does this mean?** $y_i$ is simply the "coordinate" of $\mathbf{x}$ along the direction $\phi_i$. Mathematically, you find it via the dot product: $y_i = \mathbf{x}^T \phi_i$. We haven't compressed anything yet; we've just rewritten $\mathbf{x}$ in a new coordinate system.

### Step 2: The Lower-Dimensional Approximation (Compression)
Now for the compression step. Suppose we want to compress $\mathbf{x}$ to fit on a smaller GPU. Instead of using all $n$ directions, we will only use $m$ basis vectors (where $m < n$). 

To do this, we must force the discarded coordinates (from $m+1$ all the way to $n$) to be generic, pre-selected constants $b_i$. 

$$
\hat{\mathbf{x}} = \sum_{i=1}^{m} y_i \phi_i + \sum_{i=m+1}^{n} b_i \phi_i
$$

* **Why constants?** Because we are not allowed to store the actual, varying data points $y_i$ for those dropped dimensions anymore! We have to replace them with a "best guess" constant $b_i$ that is baked into the model.

### Step 3: Defining the Representation Error (The "Oops" Factor)
Whenever you compress data, you lose information. The representation error vector $\Delta \mathbf{x}$ is the literal difference between the true vector and our compressed approximation:

$$
\Delta \mathbf{x} = \mathbf{x} - \hat{\mathbf{x}} = \sum_{i=m+1}^{n} (y_i - b_i) \phi_i
$$

We define the overall "badness" of our compression as the **Expected Mean Square Error (MSE)** of this magnitude across all samples:

$$
E[\|\Delta \mathbf{x}\|^2] = E\left[ \sum_{i=m+1}^{n} (y_i - b_i)^2 \right]
$$

*(Note: Because the basis vectors $\phi_i$ are all perfectly perpendicular, their cross-multiplied terms cleanly cancel out to 0, leaving this beautiful sum of squares).*

### Step 4: Minimizing the Error (Calculus to the Rescue)
To find the absolute smartest constants $b_i$ to use as our "best guesses", we use standard calculus: take the derivative of the error with respect to $b_i$ and set it to zero.

$$
\frac{\partial}{\partial b_i} E[\|\Delta \mathbf{x}\|^2] = -2 E[y_i - b_i] = 0 \implies b_i = E[y_i]
$$

* **The Intuition:** The math perfectly aligns with common sense! If you have to throw away a feature (like the "number of bathrooms" in a house dataset), the best constant "guess" to replace it with for every single house is simply the **average (mean)** number of bathrooms across the whole dataset.

### Step 5: The Covariance Matrix Emerges
Let's substitute our optimal guess $b_i = E[y_i]$ back into the error equation. Since we know coordinate $y_i = \mathbf{x}^T \phi_i$:

$$
E[\|\Delta \mathbf{x}\|^2] = \sum_{i=m+1}^{n} E[(y_i - E[y_i])^2] = \sum_{i=m+1}^{n} \phi_i^T E[(\mathbf{x} - E[\mathbf{x}])(\mathbf{x} - E[\mathbf{x}])^T] \phi_i
$$

Look closely at that massive central term: $E[(\mathbf{x} - E[\mathbf{x}])(\mathbf{x} - E[\mathbf{x}])^T]$. **That is the exact, literal definition of the Data Covariance Matrix ($\Sigma_x$)!** 

Thus, the entire terrifying error equation we are trying to minimize collapses beautifully to:

$$
E[\|\Delta \mathbf{x}\|^2] = \sum_{i=m+1}^{n} \phi_i^T \Sigma_x \phi_i
$$

### Step 6: Finding the Optimal Basis (Lagrange Optimization)
We now have one final task: choose the remaining spatial directions $\phi_i$ to make that error equation as close to 0 as possible. We must enforce the rule that they remain unit vectors ($\phi_i^T \phi_i = 1$). To solve constrained optimization problems, we form the Lagrangian:

$$
\mathcal{L} = \phi_i^T \Sigma_x \phi_i + \lambda_i (1 - \phi_i^T \phi_i)
$$

Taking the derivative of $\mathcal{L}$ with respect to the vector $\phi_i$ and setting it to zero yields the most famous equation in linear algebra:

$$
\Sigma_x \phi_i = \lambda_i \phi_i
$$

**The Grand Conclusion:** 
The math has spoken. The optimal basis vectors $\phi_i$ that we should project our data onto are exclusively the **eigenvectors** of the Covariance Matrix! 

Furthermore, the representation error is exactly the sum of the discarded eigenvalues $\sum_{i=m+1}^{n} \lambda_i$. To compress the data safely, we simply throw away the dimensions corresponding to the *smallest* eigenvalues (the noise), and keep the ones with the *largest* eigenvalues (the signal).

---

## 3. PCA and Singular Value Decomposition (SVD)

In modern ML frameworks, we bypass computing the massive covariance matrix entirely by directly applying SVD to the mean-centered data matrix $C \in \mathbb{R}^{n \times p}$.

### The Mathematical Link
The covariance matrix is defined as:

$$
\Sigma_C = \frac{1}{n-1} C^T C
$$

If the SVD of the centered data matrix is $C = U S V^T$, then:

$$
\Sigma_C = \frac{1}{n-1} (V S U^T)(U S V^T) = V \left( \frac{S^2}{n-1} \right) V^T
$$

Because $\Sigma_C$ is symmetric, this equation is exactly its strictly orthogonal diagonalization!

* **The Principal Directions** are perfectly captured in the columns of $V$ (the Right Singular Vectors).
* **The Eigenvalues** are precisely equal to $\frac{\sigma_i^2}{n-1}$ (where $\sigma_i$ are the singular values from the diagonal matrix $S$).
* **The Principal Components (Projected Data)** are simply $C V = U S$.

**Information Preservation Metric:**
When reducing to $K$ dimensions, the percentage of retained variance is exact:

$$
\text{Preserved Variance (\%)} = \left( \frac{\sum_{i=1}^{K} \lambda_i}{\sum_{i=1}^{p} \lambda_i} \right) \times 100
$$

---

## 4. Comprehensive Worked Examples

### Example 1: Standard 2D to 1D Pipeline
Given a simple 2D dataset of 8 samples.

* **Data Matrix:** Let $C$ be the $8 \times 2$ zero-centered matrix.
* **Covariance Matrix:** $\Sigma_C = \frac{1}{7} C^T C$.
* **Eigen Decomposition:**

$$
\Sigma_C = \begin{bmatrix} 6.25 & 4.25 \\ 4.25 & 3.5 \end{bmatrix} \implies \lambda_1 \approx 9.34, \lambda_2 \approx 0.48
$$

* **Principal Directions:** The dominant eigenvector ($\lambda_1 = 9.34$) points in the direction of $[0.83, 0.55]^T$. 
* **Projection:** Multiplying the original data by this single dominant eigenvector perfectly collapses the 2D cloud into a 1D line that retains maximum geometric spread.

### Example 2: The Importance of Zero Centering
Given a raw $10 \times 2$ coordinate matrix:

* **Zero Mean Shift:** Before any PCA operations, you must calculate the mean of $x_1$ and $x_2$, and subtract them from every sample. If you fail to center the data, the first principal component will primarily point from the origin to the center of the data mass, utterly ruining the variance analysis.
* **Extract Roots:** Suppose we extract eigenvalues $\lambda_1 = 1.28, \lambda_2 = 0.04$.
* **Dimensionality Reduction:** To reduce this to 1D, we project every original centered coordinate $(x, y)$ using the primary eigenvector $[0.67, 0.73]^T$.

$$
y_{\text{new}} = 0.67(x) + 0.73(y)
$$

Your $100$-dimensional dataset could be reduced to $20$ dimensions exactly the same way: find a $100 \times 100$ covariance matrix, select the top $20$ large eigenvectors, and project your data onto that $20$-dimensional orthogonal basis.

---

## 5. Next Steps

In solving PCA, we've developed an optimal method to reduce dimensions while maximizing variance indiscriminately. But what if we *want* to discriminate? In the next module, we will explore **Linear Discriminant Analysis (LDA)**—a technique that reduces dimensionality specifically to maximize class separation!
