## Introduction to Vector Spaces

*Essential Mathematics for ML — Structured Notes*

---

## 1. Definition of a Vector Space

### Motivation and Intuition
Why do we bother rigidly defining abstract "Vector Spaces"? Because in Machine Learning, we don't just work with geometric arrows pointing in 3D space. We work with images, audio waveforms, sentence embeddings, and probability distributions.

If we mathematically prove that a set of objects (like "all $28 \times 28$ grayscale images") qualifies as a valid Vector Space, we get a massive superpower: **Every single linear algebra theorem, optimization bound, and geometric intuition immediately applies to those images.** We can suddenly calculate the "angle" between two images, or the "length" of an audio file, entirely for free.

### Formal Definition
A **vector space** $V$ over a **field** $\mathbb{R}$ is a structure with a set of vectors, a field of scalars, and two fundamental operations:

1. **Vector Addition ($+$):** $+ : V \times V \to V$.
2. **Scalar Multiplication ($\cdot$):** $\cdot : \mathbb{R} \times V \to V$.

---

## 2. Axioms of a Vector Space

To legally be a vector space, the structure must flawlessly pass these axioms. 

### (A) Additive Properties (Abelian Group)

* **Closure:** If you add two vectors in the space, the result must stay in the space.
* **Commutativity:** $\mathbf{v} + \mathbf{w} = \mathbf{w} + \mathbf{v}$.
* **Additive Identity:** A zero vector $\mathbf{0}$ exists where $\mathbf{v} + \mathbf{0} = \mathbf{v}$.
* **Additive Inverse:** For every $\mathbf{v}$, there is a $-\mathbf{v}$ where $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$.

### (B) Multiplicative Properties

* **Scalar Closure:** Scaling a vector keeps it in the space.
* **Distributivity:** $a(\mathbf{v} + \mathbf{w}) = a\mathbf{v} + a\mathbf{w}$.

```python
import numpy as np

# A quick code proof of closure in a 2D Euclidean space
v = np.array([1, 2])
w = np.array([3, -1])

# Addition results in a 2D array, confirming closure in R^2
result = v + w  
print(result.shape) # Output: (2,)
```

---

## 3. Valid Vector Space Examples

### Euclidean Spaces ($\mathbb{R}^n$)
$\mathbb{R}^n$ is the standard environment of ML. If standard datasets have $n$ features, they live here.

### Highly Abstract Vector Spaces

* **Matrices:** The set of all $m \times n$ matrices forms a vector space. (This is why we can treat images as vectors).
* **Polynomials:** The set of all polynomials up to a certain degree.
* **Functions:** The set of all continuous functions from $\mathbb{R} \to \mathbb{R}$ is a mathematically valid vector space of infinite dimensions.

---

## 4. Non-Examples (Counter-Cases)

Where do things break? Understanding breaking points builds deep intuition.

* **Exact Degree Polynomials:** The set of polynomials of *exactly* degree 3 is NOT a vector space. 
  Example: $(2x^3) + (-2x^3 + x^2) = x^2$. 
  The new polynomial is degree 2. It "fell out" of the space. It failed **closure**.

* **First Quadrant only:** The space of 2D vectors where $x > 0, y > 0$ is NOT a vector space.
  Example: Multiply $(1, 1)$ by scalar $-1$. The result is $(-1, -1)$. It fell out of the space. It failed **scalar closure**.

---

## 5. Geometric Interpretation

### Visualizing Vectors

* **$n=1$:** A point on a number line.
* **$n=2, 3$:** Arrows originating from the origin $(0,0)$.

### Parallelogram Law of Addition
Geometrically, $\mathbf{v}_1 + \mathbf{v}_2$ forms the diagonal of a parallelogram.

### Linear Independence
In $\mathbb{R}^3$, vectors are linearly independent if they literally point in entirely distinct geometric dimensions. If a third vector lies perfectly flat on the plane defined by the first two, it provides zero new topological information—it is linearly dependent.

---

## 6. Applications in $\mathbb{R}^n$

In Deep Learning, modern models map discrete, non-math concepts into dense $\mathbb{R}^n$ vector spaces.

**The Ultimate Example: Word Embeddings (Word2Vec / Transformers)**
Words aren't numbers, but Neural Networks embed them into a 512-dimensional vector space. Because those embeddings inhabit a rigorously valid vector space, arithmetic works perfectly on concepts:

$$
\text{vec}(\text{King}) - \text{vec}(\text{Man}) + \text{vec}(\text{Woman}) \approx \text{vec}(\text{Queen})
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $\text{vec}(\cdot)$ | Vector embedding function mapping words to $\mathbb{R}^n$ | Converts discrete tokens into continuous vector representations |
| $\text{King}, \text{Man}, \text{Woman}, \text{Queen}$ | Words embedded as vectors | Example tokens demonstrating semantic arithmetic |
| $-$ | Vector subtraction | Removes the "gender" component from the King vector |
| $+$ | Vector addition | Adds the "gender" component back |
| $\approx$ | Approximate equality | Indicates the result is close to but not exactly the target |

The abstract axioms of vector spaces are precisely what allow Large Language Models to "understand" geometry in human language.

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 02A: Gaussian Elimination](Lecture%2002A%20Gaussian%20Elimination%20and%20RREF.md) — Concrete rank and null space analysis motivates the abstract framework
- **Next:** [Lecture 04: Subspaces](Lecture%2004%20Vector%20Subspace.md) — Narrows vector spaces to origin-anchored flat slices critical for dimensionality reduction
- **Related:** [Lecture 05: Basis](Lecture%2005%20Basis%20and%20Dimensions.md) — Identifies the minimal spanning set that defines a space's dimension
