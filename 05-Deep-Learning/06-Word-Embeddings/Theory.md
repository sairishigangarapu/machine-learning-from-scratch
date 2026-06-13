# Word Embeddings: From One-Hot to Meaning

## 1. The Problem with One-Hot Encoding

### Motivation and Intuition

Neural networks and most machine learning algorithms work with numbers, not words. A naive approach is to assign each word a random number, but this fails because the numeric values have no semantic meaning — similar words end up with arbitrarily different numbers. One-hot encoding solves the "arbitrary number" problem but creates sparse, high-dimensional vectors where every pair of words is equally dissimilar.

One-hot encoding represents each word as a sparse vector of length |V| (vocabulary size). For a vocabulary of 50,000 words, each word is a 50,000-dimensional vector with a single 1 and 49,999 zeros.

| Problem | Impact | Example |
| :--- | :--- | :--- |
| **No semantic similarity** | "cat" and "dog" are as different as "cat" and "democracy" | All pairs have cosine similarity = 0 (orthogonal) |
| **High dimensional** | 50K dimensions per word | Curse of dimensionality, memory explosion, slow training |
| **No compositionality** | Cannot capture analogies | "king" - "man" + "woman" is meaningless with one-hot vectors |

One-hot vectors are orthogonal by construction: for any two different words w_i and w_j, their one-hot vectors have dot product 0. This means every word is equally (dis)similar to every other word — a complete failure to capture semantics.

---

## 2. Word Embeddings: The Solution

### Motivation and Intuition

A **word embedding** maps each word to a **dense, low-dimensional vector** (e.g., 300 dimensions) where **semantically similar words are close together** in the vector space. The key insight: instead of representing a word with a binary indicator, we learn a distributed representation where each dimension captures some latent semantic feature.

| Property | One-Hot | Embedding |
| :--- | :--- | :--- |
| Representation | Sparse (mostly zeros) | Dense (all non-zero) |
| Dimensions | $\vert V\vert$ (e.g., 50,000) | 100-300 |
| Origin | Hand-designed | Learned from data |
| Semantic structure | None — all words orthogonal | Emerges from co-occurrence patterns |
| Memory for 50K words | 50,000 x 50,000 = 2.5B entries | 50,000 x 300 = 15M entries (~167x smaller) |
| Similarity | Cannot compute | Cosine similarity works |

### The Distributional Hypothesis

The foundation of all embedding methods: **"You shall know a word by the company it keeps"** (Firth, 1957). Words that appear in similar contexts have similar meanings. Embeddings learn to encode this contextual similarity into geometric proximity.

---

## 3. Word2Vec (Mikolov et al., 2013)

### Motivation and Intuition

The breakthrough paper. Word2Vec uses a shallow neural network to learn embeddings by predicting words from their context. Two architectures exist: CBOW (Continuous Bag of Words) and Skip-Gram. Both are simple, fast, and produce embeddings that capture rich semantic relationships.

### A. CBOW (Continuous Bag of Words)

Predicts a **target word** from its **context** (surrounding words). The context words are averaged (or summed) to form a single representation, which is used to predict the target.

$$
P(w_t \mid w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2})
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $P(w_t \mid \text{context})$ | Probability of the target word given context | What the model predicts — the conditional distribution over the entire vocabulary |
| $w_t$ | Target word at position $t$ | The word we are trying to predict — the "center" word |
| $w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}$ | Context words (2 before, 2 after) | The surrounding words that provide context — a window of size 2 around the target |

**CBOW Architecture:**
1. Each context word is mapped to its embedding vector (lookup table).
2. The context embeddings are averaged to produce a single "context vector."
3. The context vector is projected through a softmax to predict the target word.
4. The embeddings are updated via backpropagation.

### B. Skip-Gram

Predicts **context words** from a **target word** (inverted CBOW).

$$
P(w_{t+j} \mid w_t), \quad j \in \{-2, -1, 1, 2\}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $w_t$ | Target word at position $t$ | The input word — used to predict each surrounding context word |
| $w_{t+j}$ | Context word at offset $j$ | A word within the window ($j = \pm 1, \pm 2$) — what we predict for each position |
| $j \in \{-2, -1, 1, 2\}$ | Window offsets | The set of positions relative to the target — $j=0$ is excluded (that is the target itself) |

**Skip-Gram Architecture:**
1. The target word is mapped to its embedding vector.
2. For each context position, this vector is projected through a separate softmax (though in practice, all positions share the same output matrix).
3. The model learns to output high probability for actual context words and low probability for random words.

> **Key trade-off:** Skip-Gram works better for rare words and small datasets (it generates more training examples per occurrence). CBOW is faster and works better for frequent words.

---

## 4. Negative Sampling

### Motivation and Intuition

Computing softmax over the full vocabulary (|V| = 50,000+) is prohibitively expensive — it requires computing a dot product with every word's output vector and exponentiating each. Negative sampling reformulates the problem as binary classification: distinguish real (target, context) pairs from random (target, noise) pairs.

Instead of predicting which of 50,000 words is the right context, the model learns to answer: "is this word a valid context for that word?"

### Formal Definition

$$
\mathcal{L} = -\log \sigma(\mathbf{v}_{w_O} \cdot \mathbf{v}_{w_I}) - \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)}[\log \sigma(-\mathbf{v}_{w_i} \cdot \mathbf{v}_{w_I})]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathcal{L}$ | Total loss for one training example | Minimized; measures how well the model distinguishes real context words from noise |
| $\sigma$ | Sigmoid function $\sigma(x) = 1/(1 + e^{-x})$ | Converts dot product to a probability (0 to 1) — high dot product = high probability of being a real pair |
| $\mathbf{v}_{w_O}$ | Embedding vector of the output (context) word | The word the model should predict as valid context |
| $\mathbf{v}_{w_I}$ | Embedding vector of the input (target) word | The word used to predict context |
| $\mathbf{v}_{w_I} \cdot \mathbf{v}_{w_O}$ | Dot product between target and context embeddings | Measures similarity — positive means they should be close in vector space |
| $k$ | Number of negative samples per training step | More negatives = more robust learning, but slower (typical: 5-20) |
| $w_i \sim P_n(w)$ | A word sampled from the noise distribution | Random words that should NOT be in the context — the model learns to push these away |
| $P_n(w)$ | Noise distribution (typically $\text{unigram}^{3/4}$) | The unigram distribution raised to the 3/4 power — oversamples rare words, which are more informative as negatives |

**Intuition:** The target word's embedding should be close to context words' embeddings (high dot product -> sigmoid near 1) and far from random "noise" words (low dot product -> sigmoid near 0).

### Why Unigram^{3/4}?

Raising the unigram distribution to the 3/4 power is a heuristic: it increases the sampling probability of rare words (making them more frequent as negatives) while decreasing it for very common words (which are less informative as negatives). This produces better embeddings than uniform negative sampling.

### Negative Sampling vs Hierarchical Softmax

| Method | Approach | Complexity | When to Use |
| :--- | :--- | :--- | :--- |
| Full Softmax | exp over all | $O(\vert V\vert)$ | Small vocabularies |
| Hierarchical Softmax | Binary tree | $O(\log \vert V\vert)$ | Large vocabularies, rare words |
| Negative Sampling | Binary classification | $O(k)$ where $k \ll \vert V\vert$ | Default in Word2Vec |

---

## 5. Cosine Similarity and Embedding Arithmetic

### Motivation and Intuition

Once embeddings are learned, we measure similarity between words using **cosine similarity** — the cosine of the angle between their vectors. This is preferred over Euclidean distance because it is invariant to vector magnitude (which can vary arbitrarily during training).

The most stunning property of Word2Vec: the embedding space captures **linear relationships** between semantic concepts, enabling **semantic arithmetic**.

### Cosine Similarity

$$
\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{a} \cdot \mathbf{b}$ | Dot product of the two vectors | Measures raw alignment — how much the vectors point in the same direction |
| $\|\mathbf{a}\|$ | Magnitude (L2 norm) of vector $\mathbf{a}$ | Normalizes out vector length — cosine similarity only cares about direction |

### Semantic Analogies

$$
\begin{aligned}
\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} &\approx \vec{\text{queen}} \\
\vec{\text{Paris}} - \vec{\text{France}} + \vec{\text{Italy}} &\approx \vec{\text{Rome}} \\
\vec{\text{bigger}} - \vec{\text{big}} + \vec{\text{small}} &\approx \vec{\text{smaller}}
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\vec{\text{king}}$ | Embedding vector of "king" | The 300-dimensional dense representation learned by the model |
| $-$ (subtract $\vec{\text{man}}$) | Removes the "gender" direction | Isolates the "royalty" concept independent of gender |
| $+$ (add $\vec{\text{woman}}$) | Adds the "gender" direction back | Now applies the "royalty" concept to the female gender |
| $\approx$ (nearest neighbor) | Closest word in vector space | The closest word to the resulting vector is typically "queen" |

### Why Embedding Arithmetic Works

The vector offset man -> woman (direction from "man" to "woman" in embedding space) captures the concept of "gender." This same direction can be meaningfully applied to other word pairs (king -> queen, boy -> girl, father -> mother). The embedding space discovers these **linear semantic directions** through co-occurrence patterns in the training data.

---

## 6. GloVe (Global Vectors, Pennington et al., 2014)

### Motivation and Intuition

While Word2Vec uses local context windows, GloVe learns embeddings from the **global co-occurrence matrix** — how often words appear together across the entire corpus. This combines the benefits of count-based methods (like LSA) with prediction-based methods (like Word2Vec).

The key insight: ratios of co-occurrence probabilities encode meaning better than raw probabilities. For example, the ratio P(ice|solid) / P(steam|solid) is large (ice is solid-related), while P(ice|gas) / P(steam|gas) is small.

### Formal Definition

$$
J = \sum_{i,j=1}^{V} f(X_{ij}) (\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij})^2
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $J$ | GloVe loss function | Measures how well the dot product of two word vectors approximates their log co-occurrence |
| $V$ | Vocabulary size | The sum runs over all word pairs in the vocabulary |
| $X_{ij}$ | Co-occurrence count of words $i$ and $j$ | How many times word $j$ appears in the context of word $i$ |
| $\mathbf{w}_i$ | Embedding vector for word $i$ | The "word" vector — the final representation of word $i$ |
| $\tilde{\mathbf{w}}_j$ | Context embedding vector for word $j$ | Separate "context" vector — often averaged with $\mathbf{w}_j$ for the final embedding |
| $b_i$ | Bias for word $i$ | Allows each word to have a baseline co-occurrence level independent of the dot product |
| $\tilde{b}_j$ | Context bias for word $j$ | Same as $b_i$ but for the context word |
| $f(X_{ij})$ | Weighting function | Down-weights very frequent pairs (e.g., "the", "of") and very rare pairs |

### Word2Vec vs GloVe

| Aspect | Word2Vec | GloVe |
| :--- | :--- | :--- |
| Type | Predictive (neural network) | Count-based (matrix factorization) |
| Context | Local window (e.g., 5 words) | Global co-occurrence statistics |
| Training | Iterative SGD (fast to train) | Pre-computed statistics, fast to load |
| Performance | Comparable on most benchmarks | Comparable on most benchmarks |
| Parallelism | Easy (SGD is inherently sequential) | Easy (matrix operations) |

> In practice, both produce similar quality embeddings. GloVe is convenient when you have pre-computed co-occurrence data; Word2Vec is simpler to train on arbitrary corpora.

---

## 7. FastText (Bojanowski et al., 2017)

### Motivation and Intuition

Word2Vec and GloVe assign a single vector per word. This fails for **out-of-vocabulary (OOV) words** — words not seen during training — and performs poorly on **morphologically rich languages** where words have internal structure (e.g., "unhappiness" = "un" + "happy" + "ness").

FastText extends Word2Vec by representing each word as a **bag of character n-grams**. The word's embedding is the sum of its constituent n-gram embeddings.

### Formal Definition

$$
\text{"where"} = \{\text{wh}, \text{whe}, \text{her}, \text{ere}, \text{re}\}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\text{"where"}$ | The target word | Decomposed into character n-grams for subword information |
| $\{\text{wh}, \text{whe}, \text{her}, \text{ere}, \text{re}\}$ | Set of character 3-grams for "where" | Each n-gram gets its own embedding vector; the word's final embedding is the sum of its n-gram embeddings |

### Key Advantages

- **Handles out-of-vocabulary words:** Even if "ChatGPT" was never seen during training, its n-grams ("Cha", "hat", "atG", "tGP", "GPT") likely overlap with seen words, producing a reasonable embedding.
- **Better for morphologically rich languages:** Languages like Finnish, Turkish, German (with extensive compounding and inflection) benefit from subword information.
- **Spelling-robust:** Misspellings that share n-grams with the correct word will have similar embeddings.

---

## 8. Contextual Embeddings: ELMo -> BERT -> GPT

### Motivation and Intuition

Static embeddings (Word2Vec, GloVe, FastText) assign the **same vector** to each word regardless of context. But "bank" in "river bank" and "bank account" should be different vectors — they mean different things. This is the **polysemy problem**: one word, multiple meanings.

Contextual embeddings solve this by computing a unique vector for each occurrence of a word based on its surrounding context.

| Model | Year | Approach | Type |
| :--- | :--- | :--- | :--- |
| Word2Vec/GloVe | 2013-2014 | Static — one vector per word | Non-contextual |
| ELMo | 2018 | Contextual — vector depends on sentence (bidirectional LSTM) | Contextual |
| BERT | 2018 | Deep contextual — bidirectional transformer (masked language model) | Contextual |
| GPT | 2018-2026 | Deep contextual — unidirectional transformer (autoregressive) | Contextual |

**ELMo** (Embeddings from Language Models): Uses a bidirectional LSTM language model. The embedding for a word is a learned combination of all hidden layers, giving it access to both left and right context.

**BERT** (Bidirectional Encoder Representations from Transformers): Uses a transformer encoder with masked language modeling (predict masked words from full context). The standard for NLP tasks from 2018 onward.

**GPT** (Generative Pre-trained Transformer): Uses a transformer decoder with autoregressive language modeling (predict next word from left context). The basis for modern LLMs (GPT-3, GPT-4, etc.).

> **The journey:** Word2Vec teaches you *what embeddings are*. Transformers teach you *how contextual embeddings work*. You have seen both now.

---

## 9. Code Example: Word2Vec with Gensim

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

# ---- Training from scratch ----
sentences = [
    ["the", "king", "sat", "on", "the", "throne"],
    ["the", "queen", "sat", "on", "the", "throne"],
    ["the", "prince", "fought", "the", "dragon"],
    ["the", "princess", "fought", "the", "dragon"],
    ["the", "man", "walked", "to", "the", "market"],
    ["the", "woman", "walked", "to", "the", "market"],
    ["paris", "is", "the", "capital", "of", "france"],
    ["london", "is", "the", "capital", "of", "england"],
    ["berlin", "is", "the", "capital", "of", "germany"],
    ["rome", "is", "the", "capital", "of", "italy"],
]

model = Word2Vec(sentences, vector_size=50, window=3, min_count=1,
                 workers=1, sg=1, epochs=200, seed=42)
print(f"Vocabulary size: {len(model.wv)}")

# ---- Semantic similarity ----
print(f"king vs queen: {model.wv.similarity('king', 'queen'):.3f}")
print(f"man vs woman:  {model.wv.similarity('man', 'woman'):.3f}")

# ---- Analogy: king - man + woman = ? ----
result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=3)
print("king - man + woman =", result)

# ---- PCA / t-SNE Visualization ----
words = list(model.wv.index_to_key)
vectors = np.array([model.wv[w] for w in words])

# PCA to 2D
coords = PCA(n_components=2).fit_transform(vectors)
plt.figure(figsize=(10, 7))
for i, word in enumerate(words):
    plt.scatter(coords[i, 0], coords[i, 1], alpha=0.7, s=100)
    plt.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=9)
plt.title('Word Embeddings (PCA to 2D)')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 10. Embedding Dimension Trade-offs

| Dimensionality | Best For | Characteristics |
| :--- | :--- | :--- |
| 50-100 | Small datasets, fast training | Captures basic semantic relationships, minimal memory |
| 200-300 | General NLP tasks (sweet spot) | Good trade-off between quality and speed |
| 500-1000 | Large datasets, domain-specific tasks | Diminishing returns beyond 300 for most tasks |
| 768-4096 | Transformer models (BERT, GPT) | Contextual embeddings — much larger capacity needed |

**Rule of thumb:** For Word2Vec on most tasks, 300 dimensions is the standard choice. Larger dimensions increase the risk of overfitting on small corpora.

---

## 11. Advantages and Disadvantages

### Pros

- Capture semantic relationships that one-hot encoding cannot represent.
- Transfer learning — pre-trained embeddings boost performance on small datasets.
- Dimensionality reduction — 300D embeddings vs 50K one-hot (167x smaller).
- Enable analogy reasoning and semantic arithmetic.
- Simple and fast to train (Word2Vec training takes minutes on a laptop).

### Cons

- **Static** — same word, different meanings get the same vector (polysemy problem).
- **Context-free** — cannot handle word sense disambiguation (bank/river vs bank/money).
- Pre-trained models are biased — gender, racial, and cultural biases in training data propagate to embeddings.
- Require large corpora to train high-quality embeddings from scratch.
- Limited by the distributional hypothesis — rare words and words used in unusual contexts get poor embeddings.
- Out-of-vocabulary problem for Word2Vec/GloVe (solved by FastText and contextual models).

> **Check your intuition:** If Word2Vec embeddings capture linear relationships, what happens if you compute Paris_vec - France_vec + Italy_vec? Why does the model capture this — what property of the training data causes country-capital relationships to align along a consistent vector direction?

> **Answer:** The result should be closest to Rome_vec. The model captures this because in the training data, the relationship between a country and its capital appears consistently: "Paris" and "France" appear in similar contexts to "Rome" and "Italy" (e.g., "capital of", "is in"). The embedding space learns that the vector offset from country to capital is approximately the same direction for all country-capital pairs.

---

## Prerequisites and Further Reading

- **Prerequisites:** Neural Networks, Softmax, Cross-Entropy Loss.
- **Related:** RNNs (sequence processing), Transformers (contextual embeddings).
- **Original papers:**
  - Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (Word2Vec, 2013)
  - Pennington et al., "GloVe: Global Vectors for Word Representation" (2014)
  - Bojanowski et al., "Enriching Word Vectors with Subword Information" (FastText, 2017)
  - Peters et al., "Deep Contextualized Word Representations" (ELMo, 2018)
- **Next:** Seq2Seq and Attention Mechanisms.
