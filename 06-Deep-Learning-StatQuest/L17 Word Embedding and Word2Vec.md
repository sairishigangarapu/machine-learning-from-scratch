## Word Embedding and Word2Vec

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. Why Word Embeddings?

### Motivation and Intuition

Neural networks and most machine learning algorithms work with numbers, not words. A naive approach is to assign each word a random number, but this fails because the numeric values have no semantic meaning — similar words end up with arbitrarily different numbers. One-hot encoding solves the "arbitrary number" problem but creates sparse, high-dimensional vectors where every pair of words is equally dissimilar.

**Word embeddings** solve both problems: they map each word to a dense, low-dimensional vector where semantically similar words are close together in vector space.

### Sparse vs. Dense Representations

| Representation | Size | Semantics | Example |
| :--- | :--- | :--- | :--- |
| One-hot | $|V|$ (e.g., 50,000) | None — all words orthogonal | "cat" = [0,0,1,0,...], "dog" = [0,1,0,0,...] |
| Random | 1 | None — arbitrary values | "cat" = 0.3, "dog" = 5.1 |
| Embedding | 100-300 | Similar words have similar vectors | "cat" = [0.2, -0.1, 0.5, ...], "dog" = [0.3, -0.2, 0.4, ...] |

---

## 2. Word2Vec: Learning Embeddings with Neural Networks

### Motivation and Intuition

Word2Vec (Mikolov et al., 2013) uses a shallow neural network to learn embeddings by predicting words from their context. Two architectures exist:

### A. CBOW (Continuous Bag of Words)

Predicts the **target word** from surrounding **context words**.

$$
P(w_t \mid w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2})
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $w_t$ | Target word at position $t$ | The word being predicted |
| $w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}$ | Context words (window size 2) | Surrounding words that provide context |
| $P(w_t \mid \cdot)$ | Conditional probability over vocabulary | Model output — distribution over all words |

### B. Skip-Gram

Predicts **context words** from the **target word** (inverted CBOW).

$$
P(w_{t+j} \mid w_t), \quad j \in \{-2, -1, 1, 2\}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $w_t$ | Input target word | Used to predict surrounding words |
| $w_{t+j}$ | Context word at offset $j$ | What the model predicts for each surrounding position |

> Skip-Gram works better for rare words and small datasets. CBOW is faster.

---

## 3. Negative Sampling

### Motivation and Intuition

Computing softmax over the full vocabulary is expensive. Negative sampling reformulates the problem as binary classification: distinguish real (target, context) pairs from random (target, noise) pairs.

$$
\mathcal{L} = -\log \sigma(\mathbf{v}_{w_O} \cdot \mathbf{v}_{w_I}) - \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)}[\log \sigma(-\mathbf{v}_{w_i} \cdot \mathbf{v}_{w_I})]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\sigma$ | Sigmoid: $\sigma(x) = 1/(1+e^{-x})$ | Converts dot product to probability |
| $\mathbf{v}_{w_I}$ | Embedding of input (target) word | The word used to predict context |
| $\mathbf{v}_{w_O}$ | Embedding of output (context) word | The word predicted as context |
| $\mathbf{v}_{w_I} \cdot \mathbf{v}_{w_O}$ | Dot product similarity | Measures how related the two words are |
| $k$ | Number of negative samples per positive pair | More negatives = more robust (typical: 5-20) |
| $w_i \sim P_n(w)$ | Noise distribution (unigram$^{3/4}$) | Random words the model should learn are NOT context |

**Intuition:** Push real context pairs to have high dot product (sigmoid near 1), push random pairs to have low dot product (sigmoid near 0).

---

## 4. Cosine Similarity

### Motivation and Intuition

Once embeddings are learned, we measure similarity between words using cosine similarity — the cosine of the angle between their vectors.

$$
\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{a} \cdot \mathbf{b}$ | Dot product of the two vectors | Measures raw alignment |
| $\|\mathbf{a}\|$ | Magnitude (L2 norm) of vector $\mathbf{a}$ | Normalizes out vector length |

### Semantic Analogy Property

Word2Vec embeddings capture linear semantic relationships:

$$
\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}
$$

The vector offset $\vec{man} \to \vec{woman}$ captures the "gender" direction, which can be added to or subtracted from other word vectors.

---

## 5. Embedding Dimension

| Dimensionality | Trade-off |
| :--- | :--- |
| 50-100 | Fast, captures basic semantics |
| 200-300 | Sweet spot for most NLP tasks |
| 500+ | Diminishing returns, slower |

---

## 6. Python Code: Word2Vec with Gensim

```python
import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load pre-trained Word2Vec (Google News, 300d)
model = api.load('word2vec-google-news-300')

# Semantic analogies
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=3)
print("king - man + woman =", result)

# Similarity scores
print(f"cat vs dog: {model.similarity('cat', 'dog'):.3f}")
print(f"cat vs car: {model.similarity('cat', 'car'):.3f}")

# Find nearest neighbors
print("Words most similar to 'neural':")
print(model.most_similar('neural', topn=5))

# Visualize embeddings in 2D with PCA
words = ['king', 'queen', 'man', 'woman', 'prince', 'princess',
         'paris', 'france', 'london', 'england', 'berlin', 'germany']
vectors = np.array([model[w] for w in words])
coords = PCA(n_components=2).fit_transform(vectors)

plt.figure(figsize=(10, 7))
plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6)
for i, word in enumerate(words):
    plt.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=12)
plt.title('Word Embeddings Visualized with PCA (2D)')
plt.grid(True)
plt.show()
```

---

> **Check your intuition:** If Word2Vec embeddings capture linear relationships, what happens if you compute $\vec{Paris} - \vec{France} + \vec{Italy}$? Why does the model capture this — what property of the training data causes country-capital relationships to align along a consistent vector direction?

---

## Prerequisites and Further Reading

- [StatQuest: Neural Networks](https://www.youtube.com/watch?v=CqOfi41LfDw)
- [StatQuest: Backpropagation](https://www.youtube.com/watch?v=i94OvYb6noo)
- [StatQuest: Softmax Function](https://www.youtube.com/watch?v=8tE2v4f3K-s)
- [StatQuest: Cross Entropy](https://www.youtube.com/watch?v=6ArSys5qHAU)
- Original Word2Vec paper: Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013)
- Gensim: Topic modelling for humans — https://radimrehurek.com/gensim/
