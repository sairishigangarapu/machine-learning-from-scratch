## L32: Word Embedding in PyTorch + Lightning

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. What is an Embedding?

### Motivation and Intuition

Neural networks cannot process raw text. We must convert words (or tokens) into numerical vectors. A **one-hot encoding** creates sparse vectors of size $V$ (vocabulary size) with a single 1. This is wasteful — $V$ can be 50k or more.

An **embedding** maps each token to a dense, low-dimensional vector (e.g., 100–768 dimensions). The embedding matrix $\mathbf{E} \in \mathbb{R}^{V \times d}$ is learned during training. Words with similar meanings end up with similar vectors.

$$
\text{embedding}(\text{word}_i) = \mathbf{E}[i, :]
$$

In PyTorch:

```python
import torch.nn as nn

vocab_size = 10000
embedding_dim = 128
embed = nn.Embedding(vocab_size, embedding_dim)

# Input: tensor of token indices, shape (batch, seq_len)
tokens = torch.randint(0, vocab_size, (32, 20))
embedded = embed(tokens)          # shape (32, 20, 128)
```

| Property | One-Hot Encoding | Embedding |
| :--- | :--- | :--- |
| Vector size | $V$ (e.g., 50,000) | $d$ (e.g., 128) |
| Sparsity | Sparse (one 1, rest 0) | Dense |
| Semantic similarity | Not captured | Captured (similar words near each other) |
| Learned | No | Yes, via backpropagation |
| Memory | $O(V)$ per sample | $O(V \times d)$ total |

---

## 2. Training Embeddings from Scratch

### Motivation and Intuition

Embeddings can be trained as part of any neural network. For a simple classifier, the embedding layer is just the first layer, and its weights are updated via backpropagation just like any other layer.

```python
import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim

class TextClassifier(L.LightningModule):
    def __init__(self, vocab_size, embedding_dim, num_classes, max_len):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(max_len * embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # x: (batch, seq_len) of token indices
        emb = self.embedding(x)          # (batch, seq_len, embedding_dim)
        return self.fc(emb)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
```

The embedding matrix learns to position semantically similar tokens close together in the vector space.

---

## 3. Using Pretrained Embeddings

### Motivation and Intuition

Training embeddings from scratch requires large datasets. Pretrained embeddings (Word2Vec, GloVe, FastText) were trained on billions of tokens and capture rich semantic relationships. You can either:

1. **Freeze** them and use as fixed features.
2. **Fine-tune** them by continuing gradient updates.

```python
import torch
import torch.nn as nn

# Load pretrained GloVe vectors (using torchtext)
from torchtext.vocab import GloVe

glove = GloVe(name="6B", dim=100)   # 6 billion tokens, 100d vectors
print(glove["king"])                 # tensor of shape (100,)

# Build embedding matrix from GloVe
vocab = ["<pad>", "<unk>", "king", "queen", "man", "woman"]
embedding_dim = 100
weights = torch.randn(len(vocab), embedding_dim)

for i, word in enumerate(vocab):
    if word in glove.stoi:
        weights[i] = glove[word]

# Create embedding layer with pretrained weights
embed = nn.Embedding.from_pretrained(weights, freeze=True)   # frozen
embed_finetune = nn.Embedding.from_pretrained(weights, freeze=False)  # trainable
```

| Strategy | Use Case | Effect |
| :--- | :--- | :--- |
| `freeze=True` | Small dataset, transfer learning | Pretrained knowledge preserved |
| `freeze=False` | Large dataset, domain adaptation | Embeddings adapt to the task |
| Random init + train | No pretrained available | Learn everything from scratch |

---

## 4. Embedding Visualization with TensorBoard

### Motivation and Intuition

Visualizing high-dimensional embeddings helps verify they capture meaningful relationships. TensorBoard's embedding projector uses PCA or t-SNE to project vectors into 2D/3D.

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/embeddings")

# Get embedding weights
embedding_matrix = model.embedding.weight.detach().cpu()  # (vocab_size, emb_dim)

# Metadata: labels for each token
metadata = ["<pad>", "<unk>", "king", "queen", "man", "woman"]

writer.add_embedding(
    embedding_matrix,
    metadata=metadata,
    tag="word_embeddings",
    global_step=0,
)
writer.close()
```

To view: `tensorboard --logdir=runs` then navigate to the PROJECTOR tab.

### With Lightning Logging

```python
class EmbeddingVisualizer(L.LightningModule):
    def on_train_end(self):
        # Called automatically after training
        if self.logger is not None:
            emb = self.embedding.weight.data.cpu()
            self.logger.experiment.add_embedding(
                emb, metadata=self.vocab, global_step=self.current_epoch
            )
```

---

## 5. Word2Vec-Style Embedding Training

### Motivation and Intuition

The classic Word2Vec approach uses a simple neural network to learn embeddings by predicting a word from its context (CBOW) or the context from a word (skip-gram).

Skip-gram objective: given a center word, predict surrounding context words.

```python
class SkipGram(L.LightningModule):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.save_hyperparameters()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_words, context_words):
        # center_words: (batch,)
        # context_words: (batch,)
        center_emb = self.in_embed(center_words)       # (batch, emb_dim)
        context_emb = self.out_embed(context_words)     # (batch, emb_dim)
        scores = (center_emb * context_emb).sum(dim=1)  # dot product
        return scores

    def training_step(self, batch, batch_idx):
        center, context, labels = batch
        scores = self(center, context)
        loss = nn.functional.binary_cross_entropy_with_logits(scores, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
```

---

## 6. Using nn.Embedding with an LSTM

### Motivation and Intuition

The most common pattern: embed tokens, then process with an LSTM/transformer.

```python
class EmbeddingLSTM(L.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) token indices
        emb = self.embedding(x)            # (batch, seq_len, emb_dim)
        out, _ = self.lstm(emb)            # (batch, seq_len, hidden)
        return self.classifier(out[:, -1, :])

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
```

---

> **Check your intuition:** Why does freezing pretrained embeddings make sense when you have a very small labeled dataset? When would you want to fine-tune them instead?

---

## Prerequisites and Further Reading

- **StatQuest:** Word Embedding and Word2Vec (L17), LSTM with PyTorch + Lightning (L31), Introduction to PyTorch (L29)
- **Tools:** TensorBoard embedding projector, gensim for Word2Vec
- **Papers:** Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013); Pennington et al., "GloVe: Global Vectors for Word Representation" (2014)
