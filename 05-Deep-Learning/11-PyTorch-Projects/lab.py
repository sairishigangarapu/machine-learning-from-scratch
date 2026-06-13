"""
11-PyTorch-Projects/lab.py
Comprehensive PyTorch lab covering tensor ops, autograd, NN training,
LSTM, embeddings, and a mini transformer for text generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np

# ============================================================
# PART 1: Tensor Operations and Autograd Demo
# ============================================================
print("=" * 60)
print("PART 1: Tensor Operations and Autograd")
print("=" * 60)

# Create tensors
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

print(f"x shape: {x.shape}, y shape: {y.shape}")
print(f"x + y =\n{x + y}")
print(f"x * y =\n{x * y}")
print(f"x @ y =\n{x @ y}")  # matrix multiply

# Broadcasting
a = torch.tensor([[1.0], [2.0], [3.0]])
b = torch.tensor([10.0, 20.0, 30.0])
print(f"\nBroadcasting: ({a.shape}) + ({b.shape}) = {(a + b).shape}")
print(f"Result:\n{a + b}")

# Autograd
x_grad = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y_grad = x_grad ** 2 + 3 * x_grad
z_grad = y_grad.sum()
z_grad.backward()
print(f"\nAutograd: x={x_grad.detach().numpy()}, dz/dx={x_grad.grad.numpy()}")

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# PART 2: Simple Neural Network -- Regression
# ============================================================
print("\n" + "=" * 60)
print("PART 2: Neural Network Regression from Scratch")
print("=" * 60)

# Synthetic data: y = 3*sin(x) + noise
X_reg = torch.linspace(-3, 3, 200).reshape(-1, 1)
y_reg = 3 * torch.sin(X_reg) + 0.3 * torch.randn_like(X_reg)

class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)

model_reg = RegressionNet()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model_reg.parameters(), lr=0.01)

for epoch in range(500):
    y_pred = model_reg(X_reg)
    loss = loss_fn(y_pred, y_reg)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d}, loss = {loss.item():.6f}")

with torch.no_grad():
    test_x = torch.tensor([[0.0], [1.5], [-1.5]])
    test_y = model_reg(test_x)
    print(f"Prediction at x=0:   {test_y[0].item():.4f} (expected ~0.0)")
    print(f"Prediction at x=1.5: {test_y[1].item():.4f} (expected ~3.0)")
    print(f"Prediction at x=-1.5:{test_y[2].item():.4f} (expected ~-3.0)")

# ============================================================
# PART 3: LSTM for Sine Wave Prediction
# ============================================================
print("\n" + "=" * 60)
print("PART 3: LSTM for Sine Wave Prediction")
print("=" * 60)

# Generate sine wave data
seq_len = 20
t = np.linspace(0, 40, 800)
data = np.sin(t) + 0.05 * np.random.randn(800)
data_t = torch.tensor(data, dtype=torch.float32).view(-1, 1)

class SineDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return x, y

dataset = SineDataset(data_t, seq_len)
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model_lstm = SimpleLSTM()
loss_fn_lstm = nn.MSELoss()
optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=0.01)

for epoch in range(100):
    train_loss = 0.0
    for x_batch, y_batch in train_dl:
        y_pred = model_lstm(x_batch)
        loss = loss_fn_lstm(y_pred, y_batch)
        optimizer_lstm.zero_grad()
        loss.backward()
        optimizer_lstm.step()
        train_loss += loss.item()

    if epoch % 20 == 19:
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_dl:
                y_pred = model_lstm(x_batch)
                val_loss += loss_fn_lstm(y_pred, y_batch).item()
        print(f"Epoch {epoch+1:3d}, train loss={train_loss/len(train_dl):.6f}, "
              f"val loss={val_loss/len(val_dl):.6f}")

# Autoregressive prediction
print("\nAutoregressive prediction (10 steps):")
model_lstm.eval()
seed = data_t[:seq_len].unsqueeze(0)  # (1, 20, 1)
with torch.no_grad():
    future_preds = []
    current_seq = seed.clone()
    for step in range(10):
        pred = model_lstm(current_seq)
        future_preds.append(pred.item())
        current_seq = torch.cat([current_seq[:, 1:, :], pred.unsqueeze(1)], dim=1)
    print(f"  True next: {data_t[seq_len:seq_len+10].squeeze().numpy().round(3)}")
    print(f"  Predicted: {np.array(future_preds).round(3)}")

# ============================================================
# PART 4: nn.Embedding and Word Similarity
# ============================================================
print("\n" + "=" * 60)
print("PART 4: nn.Embedding and Word Similarity")
print("=" * 60)

vocab_size = 50
emb_dim = 16
embed = nn.Embedding(vocab_size, emb_dim)

# Create synthetic token sequences
tokens = torch.randint(0, vocab_size, (4, 6))  # (batch, seq_len)
embedded = embed(tokens)
print(f"Token indices: {tokens.shape}")
print(f"Embedded: {embedded.shape}")

# Word similarity: compare embedding vectors
print("\nCosine similarity between learned embeddings:")
word_a = embed(torch.tensor([5]))   # (1, emb_dim)
word_b = embed(torch.tensor([12]))  # (1, emb_dim)
word_c = embed(torch.tensor([5]))   # same as word_a
sim_ab = F.cosine_similarity(word_a, word_b)
sim_ac = F.cosine_similarity(word_a, word_c)
print(f"  sim(token_5, token_12) = {sim_ab.item():.4f}")
print(f"  sim(token_5, token_5)  = {sim_ac.item():.4f}")

# Embedding as first layer of classifier
class EmbeddingClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_len, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.fc = nn.Linear(max_len * emb_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)                 # (B, T, D)
        return self.fc(emb.view(x.size(0), -1))  # flatten

model_emb = EmbeddingClassifier(vocab_size, emb_dim, max_len=6)
sample_out = model_emb(tokens)
print(f"Embedding classifier output shape: {sample_out.shape}")

# ============================================================
# PART 5: Mini Transformer for Text Generation
# ============================================================
print("\n" + "=" * 60)
print("PART 5: Mini Transformer (Character-Level Text Generation)")
print("=" * 60)

# --- Tokenizer ---
text = "hello world this is a character level transformer for text generation " * 50
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")

def encode(s):
    return [stoi[ch] for ch in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

# --- Dataset ---
block_size = 32
data_ids = torch.tensor(encode(text), dtype=torch.long)

class CharDataset(Dataset):
    def __init__(self, ids, block_size):
        self.ids = ids
        self.block_size = block_size

    def __len__(self):
        return len(self.ids) - self.block_size

    def __getitem__(self, idx):
        x = self.ids[idx:idx + self.block_size]
        y = self.ids[idx + 1:idx + self.block_size + 1]
        return x, y

ds = CharDataset(data_ids, block_size)
dl = DataLoader(ds, batch_size=32, shuffle=True)

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# --- Causal Self-Attention ---
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

    def forward(self, x):
        B, N, D = x.shape
        Q = self.W_Q(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        A = F.softmax(scores, dim=-1)
        context = A @ V
        out = context.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_O(out)

# --- Decoder Block ---
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

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

# --- Mini Decoder-Only Transformer ---
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=3,
                 d_ff=256, max_len=128, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)

# --- Training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_trans = MiniTransformer(vocab_size).to(device)
loss_fn_trans = nn.CrossEntropyLoss()
optimizer_trans = optim.Adam(model_trans.parameters(), lr=0.003)

print("\nTraining mini transformer for text generation...")
for epoch in range(20):
    total_loss = 0.0
    for x_batch, y_batch in dl:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model_trans(x_batch)                         # (B, N, V)
        loss = loss_fn_trans(logits.view(-1, vocab_size), y_batch.view(-1))
        optimizer_trans.zero_grad()
        loss.backward()
        optimizer_trans.step()
        total_loss += loss.item()
    if epoch % 5 == 4:
        print(f"  Epoch {epoch+1:2d}, loss={total_loss/len(dl):.4f}")

# --- Generation ---
print("\nGenerated text:")
model_trans.eval()
prompt = "hello world"
with torch.no_grad():
    gen_ids = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(100):
        logits = model_trans(gen_ids[:, -block_size:])[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        gen_ids = torch.cat([gen_ids, torch.tensor([[next_id]], device=device)], dim=1)
    generated = decode(gen_ids[0].tolist())
    print(f"  Prompt: '{prompt}'")
    print(f"  Generated:\n  {generated}")

# ============================================================
# PART 6: Best Practices Demo -- DataLoader and Checkpoint
# ============================================================
print("\n" + "=" * 60)
print("PART 6: Best Practices Demo")
print("=" * 60)

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device management
print(f"  Using device: {device}")
print(f"  CUDA available: {torch.cuda.is_available()}")

# Model summary
total_params = sum(p.numel() for p in model_trans.parameters())
print(f"  MiniTransformer parameters: {total_params:,}")

# Gradient flow check
for name, param in model_trans.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm == 0:
            print(f"  WARNING: {name} has zero gradient!")
        break  # just check first non-None

print("\nAll lab parts completed successfully.")
