# Word Embeddings: From One-Hot to Meaning

## 1. The Problem with One-Hot Encoding

One-hot encoding represents words as sparse vectors of length $|V|$ (vocabulary size). For a vocabulary of 50,000 words, each word is a 50,000-dimensional vector with a single 1 and 49,999 zeros.

| Problem | Impact |
| :--- | :--- |
| **No semantic similarity** | "cat" and "dog" are as different as "cat" and "democracy" (equal distance) |
| **High dimensional** | 50K dimensions per word — wasteful |
| **No compositionality** | Can't capture that "king" - "man" + "woman" ≈ "queen" |

---

## 2. Word Embeddings: The Solution

A **word embedding** maps each word to a **dense, low-dimensional vector** (e.g., 300 dimensions) where **semantically similar words are close together** in the vector space.

| One-Hot | Embedding |
| :--- | :--- |
| Sparse (mostly zeros) | Dense (all non-zero) |
| Fixed | Learned from data |
| No semantic structure | Semantic structure emerges |
| 50,000 dimensions | 100–300 dimensions |

---

## 3. Word2Vec (Mikolov et al., 2013)

The breakthrough paper. Two architectures:

### A. CBOW (Continuous Bag of Words)
Predicts a **target word** from its **context** (surrounding words).

$$
P(w_t \mid w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2})
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $P(w_t \mid \cdot)$ | Probability of the target word given context | What the model predicts — the conditional distribution over the vocabulary |
| $w_t$ | Target word at position $t$ | The word we're trying to predict |
| $w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}$ | Context words (2 before, 2 after) | The surrounding words that provide context — a window of size 2 |

### B. Skip-Gram
Predicts **context words** from a **target word** (inverted CBOW).

$$
P(w_{t+j} \mid w_t), \quad j \in \{-2, -1, 1, 2\}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $P(w_{t+j} \mid w_t)$ | Probability of a context word given the target | What the model predicts — for each target word, it predicts each surrounding word |
| $w_t$ | Target word at position $t$ | The input word — used to predict context |
| $w_{t+j}$ | Context word at offset $j$ | A word within the window ($j = \pm 1, \pm 2$) |
| $j \in \{-2, -1, 1, 2\}$ | Window offsets | The set of positions relative to the target — $j=0$ is excluded (that's the target itself) |

> Skip-Gram works better for rare words and small datasets. CBOW is faster.

### The Training Trick: Negative Sampling
Instead of computing softmax over the entire vocabulary (expensive), sample a few "negative" words and train a binary classifier:

$$
\mathcal{L} = -\log \sigma(\mathbf{v}_{w_O} \cdot \mathbf{v}_{w_I}) - \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)}[\log \sigma(-\mathbf{v}_{w_i} \cdot \mathbf{v}_{w_I})]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathcal{L}$ | Total loss for one training example | We minimize this — it measures how well the model distinguishes real context words from noise |
| $\sigma$ | Sigmoid function $\sigma(x) = \frac{1}{1+e^{-x}}$ | Converts dot product to a probability (0 to 1) — high dot product = high probability of being a real pair |
| $\mathbf{v}_{w_O}$ | Embedding vector of the output (context) word | The word we want the model to predict as context |
| $\mathbf{v}_{w_I}$ | Embedding vector of the input (target) word | The word we're using to predict context |
| $\mathbf{v}_{w_I} \cdot \mathbf{v}_{w_O}$ | Dot product between target and context embeddings | Measures similarity — positive means they should be close in vector space |
| $k$ | Number of negative samples per training step | More negatives = more robust learning, but slower training |
| $w_i \sim P_n(w)$ | A word sampled from the noise distribution | Random words that should NOT be in the context — the model learns to push these away |
| $P_n(w)$ | Noise distribution (typically unigram$^{3/4}$) | The unigram distribution raised to the 3/4 power — oversamples rare words, which are more informative as negatives |

**Intuition:** The target word's embedding should be close to context words' embeddings and far from random "noise" words.

---

## 4. Famous Analogies

The most stunning property of Word2Vec: **semantic arithmetic**.

$$
\begin{aligned}
\vec{king} - \vec{man} + \vec{woman} &\approx \vec{queen} \\
\vec{Paris} - \vec{France} + \vec{Italy} &\approx \vec{Rome} \\
\vec{bigger} - \vec{big} + \vec{small} &\approx \vec{smaller}
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\vec{king}$ | Embedding vector of "king" | The 300-dimensional dense representation learned by the model |
| $-$ | Vector subtraction | Removes the "gender" direction from the king vector, leaving the "royalty" concept |
| $+$ | Vector addition | Adds the "gender" direction back, but now starting from "queen" |
| $\approx$ | Approximate equality in vector space | Not exact — the nearest neighbor is "queen" after the arithmetic |
| Nearest neighbor search | Finding the closest word in embedding space | After the arithmetic, we look up which word is closest to the resulting vector |

This works because the embedding space captures **linear relationships** between semantic concepts.

---

## 5. GloVe (Global Vectors, Pennington et al., 2014)

GloVe learns embeddings from the **co-occurrence matrix** (how often words appear together).

$$
J = \sum_{i,j=1}^{V} f(X_{ij})(\mathbf{w}_i^T\tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij})^2
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $J$ | GloVe loss function | We minimize this — measures how well the dot product of two word vectors approximates their log co-occurrence |
| $V$ | Vocabulary size | The sum runs over all word pairs in the vocabulary |
| $X_{ij}$ | Co-occurrence count of words $i$ and $j$ | How many times word $j$ appears in the context of word $i$ (or vice versa) |
| $\mathbf{w}_i$ | Embedding vector for word $i$ | The "word" vector — the final representation of word $i$ |
| $\tilde{\mathbf{w}}_j$ | Context embedding vector for word $j$ | Separate "context" vector — in practice, often averaged with $\mathbf{w}_j$ for the final embedding |
| $b_i$ | Bias for word $i$ | Allows each word to have a baseline co-occurrence level independent of the dot product |
| $\tilde{b}_j$ | Context bias for word $j$ | Same as $b_i$ but for the context word |
| $f(X_{ij})$ | Weighting function | Down-weights very frequent pairs (e.g., "the", "of") and very rare pairs — prevents them from dominating the loss |

| Word2Vec | GloVe |
| :--- | :--- |
| Predictive (neural network) | Count-based (matrix factorization) |
| Local context window | Global co-occurrence statistics |
| Fast to train | Pre-computed, fast to load |

> In practice, both produce similar quality. GloVe is convenient when you have pre-computed co-occurrence data.

---

## 6. FastText (Bojanowski et al., 2017)

Extends Word2Vec by representing words as **bags of character n-grams**:

$$
\text{"where"} = \{wh, whe, her, ere, re\}
$$

* Handles **out-of-vocabulary** words (can embed misspelled words).
* Better for **morphologically rich** languages (agglutinative, compounding).

---

## 7. Code Example

```python
import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load pre-trained Word2Vec (Google News, 300d)
model = api.load('word2vec-google-news-300')

# Analogy: king - man + woman = ?
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=3)
print("king - man + woman =", result) # [('queen', 0.71), ...]

# Semantic similarity
print(f"cat-dog: {model.similarity('cat', 'dog'):.3f}")
print(f"cat-car: {model.similarity('cat', 'car'):.3f}")

# Visualize in 2D
words = ['king', 'queen', 'man', 'woman', 'prince', 'princess',
 'Paris', 'France', 'London', 'England', 'Berlin', 'Germany']
vectors = np.array([model[w] for w in words])
coords = PCA(n_components=2).fit_transform(vectors)

plt.figure(figsize=(10, 7))
plt.scatter(coords[:, 0], coords[:, 1], c='blue', alpha=0.5)
for i, word in enumerate(words):
 plt.annotate(word, (coords[i, 0], coords[i, 1]))
plt.title('Word Embeddings (PCA to 2D)')
plt.grid(True)
plt.show()
```

---

## 8. Contextual Embeddings: ELMo → BERT → GPT

Static embeddings (Word2Vec, GloVe) assign the **same vector** to each word regardless of context. But "bank" in "river bank" and "bank account" should be different.

| Model | Year | Approach |
| :--- | :--- | :--- |
| **Word2Vec/GloVe** | 2013-2014 | Static — one vector per word |
| **ELMo** | 2018 | Contextual — vector depends on sentence (LSTM-based) |
| **BERT** | 2018 | Deep contextual — bidirectional transformer |
| **GPT** | 2018-2026 | Deep contextual — unidirectional transformer |

> **The journey:** Word2Vec teaches you *what embeddings are*. Transformers teach you *how contextual embeddings work*. You've seen both now.

---

## 9. Embedding Dimension

| Dimensionality | Trade-off |
| :--- | :--- |
| 50–100 | Fast, captures basic semantics |
| 200–300 | Sweet spot for most NLP tasks |
| 500–1000 | Richer but slower, diminishing returns |
| 768–4096 | Transformer hidden sizes (contextual) |

---

## 10. Advantages & Disadvantages

### Pros
* Capture semantic relationships that one-hot encoding cannot.
* Transfer learning — pre-trained embeddings boost small datasets.
* Dimensionality reduction — 300D embeddings vs 50K one-hot.
* Enable analogy reasoning and semantic arithmetic.

### Cons
* **Static** — same word, different meanings get the same vector.
* **Context-free** — can't handle polysemy (bank/river bank vs bank/money).
* Pre-trained models are biased (gender, racial biases in training data).
* Require large corpora to train from scratch.

---

**Previous:** [Transformers](../TRANSFORMERS/Theory.md) | **Related:** [RNN](../RNN/Theory.md) | **Related:** [ANN](../../02-Supervised-Learning/ARTIFICIAL%20NEURAL%20NETWORKS/Theory.md)
