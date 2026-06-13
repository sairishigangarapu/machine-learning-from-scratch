## L33: Coding ChatGPT Like Transformer From Scratch

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. Architecture Overview

### Motivation and Intuition

ChatGPT uses a **decoder-only transformer**: a stack of transformer blocks with causal (masked) self-attention. Each block processes the entire sequence but prevents tokens from attending to future positions.

The architecture:

1. **Token Embedding** — maps token IDs to dense vectors
2. **Positional Encoding** — adds position information
3. **Stack of Decoder Blocks**, each with:
   - Causal multi-head self-attention
   - Residual connection + layer norm
   - Feed-forward network (MLP)
   - Residual connection + layer norm
4. **Output projection** — maps hidden states to vocabulary logits

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torch.optim as optim
```

---

## 2. Data Preparation: Tokenization and Vocabulary

### Motivation and Intuition

Raw text must be converted to integer token IDs. For simplicity, we use a character-level vocabulary.

```python
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

# Example
text = "StatQuest is awesome!"
tokenizer = CharTokenizer(text)
ids = tokenizer.encode(text)     # [11, 12,  1, ...]
decoded = tokenizer.decode(ids)  # "StatQuest is awesome!"
```

For training, we create input–target pairs: given tokens $[t_1, t_2, \dots, t_n]$, predict $[t_2, t_3, \dots, t_{n+1}]$.

```python
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, text, seq_length, tokenizer):
        self.ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.ids) - self.seq_length

    def __getitem__(self, idx):
        x = self.ids[idx:idx + self.seq_length]
        y = self.ids[idx + 1:idx + self.seq_length + 1]
        return x, y
```

---

## 3. Positional Encoding

### Motivation and Intuition

Self-attention is permutation-invariant — it has no inherent notion of token order. We add positional information via sinusoidal encodings (or learned embeddings).

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
\quad
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)     # saved, not trained

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:x.size(1), :]
```

---

## 4. Causal Self-Attention

### Motivation and Intuition

Each token can only attend to itself and previous tokens. We implement this with a triangular mask: future positions are set to $-\infty$ before softmax.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        Q = self.W_Q(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)

        if mask is None:
            mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        A = F.softmax(scores, dim=-1)
        context = A @ V                          # (B, H, N, d_k)

        out = context.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_O(out)
```

---

## 5. Decoder Block

### Motivation and Intuition

A single decoder block: attention → residual + norm → MLP → residual + norm.

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = CausalSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x
```

---

## 6. Full Decoder-Only Transformer

### Motivation and Intuition

Stack the embedding, positional encoding, and decoder blocks. The final linear layer projects hidden states to vocabulary size.

```python
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)   # language model head

    def forward(self, x):
        # x: (batch, seq_len) token IDs
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1).bool()
        x = self.token_embedding(x)          # (B, N, d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        logits = self.lm_head(x)             # (B, N, vocab_size)
        return logits
```

---

## 7. LightningModule for Training

### Motivation and Intuition

Wrapping in Lightning gives us the training loop, checkpointing, and GPU support.

```python
class TransformerLM(L.LightningModule):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=4,
                 d_ff=512, max_len=128, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = DecoderOnlyTransformer(
            vocab_size, d_model, num_heads, num_layers, d_ff, max_len
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)                        # (B, N, V)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    def generate(self, prompt_ids, max_new_tokens=50):
        self.eval()
        with torch.no_grad():
            ids = prompt_ids.clone()
            for _ in range(max_new_tokens):
                logits = self(ids[:, -self.hparams.max_len:])  # crop to max_len
                next_logits = logits[0, -1, :]                  # last token logits
                next_id = torch.multinomial(F.softmax(next_logits, dim=-1), 1)
                ids = torch.cat([ids, next_id], dim=1)
        return ids
```

---

## 8. Text Generation Loop

### Motivation and Intuition

Autoregressive generation: feed the prompt, sample the next token, append it to the sequence, and repeat.

```python
def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=1.0):
    model.eval()
    ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(ids)[0, -1, :]           # (vocab_size,)
            logits = logits / temperature            # temperature scaling
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)

    return tokenizer.decode(ids[0].tolist())
```

### Temperature Effect

| Temperature | Behavior |
| :--- | :--- |
| $T \to 0$ | Greedy — always pick the most likely token, repetitive |
| $T = 1.0$ | Normal sampling — balanced creativity |
| $T > 1$ | Higher entropy — more diverse, may become incoherent |

---

## 9. Full Training Example

```python
# Data
text = "StatQuest is awesome. " * 1000
tokenizer = CharTokenizer(text)
dataset = TextDataset(text, seq_length=32, tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Model
model = TransformerLM(
    vocab_size=tokenizer.vocab_size,
    d_model=128, num_heads=4, num_layers=4, d_ff=512, max_len=32
)

# Training
trainer = L.Trainer(max_epochs=100, accelerator="auto")
trainer.fit(model, dataloader)

# Generate
prompt = "StatQuest"
generated = generate_text(model.model, tokenizer, prompt, max_tokens=50, temperature=0.8)
print(generated)
```

---

> **Check your intuition:** Why must the attention mask be triangular? What would happen if we used bidirectional (full) attention in a decoder-only model during generation?

---

## Prerequisites and Further Reading

- **StatQuest:** Decoder-Only Transformers (L21), Matrix Math Behind Transformer NN (L28), Introduction to PyTorch (L29), Coding NN with Lightning (L30)
- **Paper:** Vaswani et al., "Attention Is All You Need" (2017); Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT, 2018)
- **Concepts:** Autoregressive generation, causal masking, softmax temperature, tokenization
