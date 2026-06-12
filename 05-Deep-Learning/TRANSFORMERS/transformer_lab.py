import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
 def __init__(self, embed_size, num_heads):
 super().__init__()
 self.num_heads = num_heads
 self.head_dim = embed_size // num_heads
 assert embed_size % num_heads == 0

 self.W_q = nn.Linear(embed_size, embed_size)
 self.W_k = nn.Linear(embed_size, embed_size)
 self.W_v = nn.Linear(embed_size, embed_size)
 self.W_o = nn.Linear(embed_size, embed_size)

 def forward(self, x, mask=None):
 B, T, C = x.shape
 q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
 k = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
 v = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

 scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
 if mask is not None:
 scores = scores.masked_fill(mask == 0, float('-inf'))
 attn = F.softmax(scores, dim=-1)
 out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
 return self.W_o(out)

class TransformerBlock(nn.Module):
 def __init__(self, embed_size, num_heads, ff_dim, dropout=0.1):
 super().__init__()
 self.attn = SelfAttention(embed_size, num_heads)
 self.ff = nn.Sequential(
 nn.Linear(embed_size, ff_dim),
 nn.GELU(),
 nn.Linear(ff_dim, embed_size),
 )
 self.norm1 = nn.LayerNorm(embed_size)
 self.norm2 = nn.LayerNorm(embed_size)
 self.drop = nn.Dropout(dropout)

 def forward(self, x, mask=None):
 x = x + self.drop(self.attn(self.norm1(x), mask))
 x = x + self.drop(self.ff(self.norm2(x)))
 return x

class MiniGPT(nn.Module):
 def __init__(self, vocab_size, embed_size=128, num_heads=4, num_layers=4, block_size=64):
 super().__init__()
 self.token_emb = nn.Embedding(vocab_size, embed_size)
 self.pos_emb = nn.Embedding(block_size, embed_size)
 self.blocks = nn.Sequential(*[
 TransformerBlock(embed_size, num_heads, embed_size * 4)
 for _ in range(num_layers)
 ])
 self.norm = nn.LayerNorm(embed_size)
 self.head = nn.Linear(embed_size, vocab_size)
 self.block_size = block_size

 def forward(self, idx, targets=None):
 B, T = idx.shape
 tok_emb = self.token_emb(idx)
 pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
 x = self.blocks(tok_emb + pos_emb)
 x = self.norm(x)
 logits = self.head(x)

 loss = None
 if targets is not None:
 loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
 return logits, loss

def run_transformer_lab():
 # ---------------------------------------------------------
 # 1. Self-Attention Demo
 # ---------------------------------------------------------
 print("=" * 50)
 print("PART 1: Self-Attention Visualization")
 print("=" * 50)

 torch.manual_seed(42)
 attn = SelfAttention(embed_size=64, num_heads=8)

 # Simulate a sequence: "The cat sat on the mat"
 tokens = ["The", "cat", "sat", "on", "the", "mat"]
 x = torch.randn(1, len(tokens), 64)

 # Get attention weights
 with torch.no_grad():
 q = attn.W_q(x).view(1, -1, 8, 8).transpose(1, 2)
 k = attn.W_k(x).view(1, -1, 8, 8).transpose(1, 2)
 scores = (q @ k.transpose(-2, -1)) / (8 ** 0.5)
 weights = F.softmax(scores, dim=-1)

 # Average across heads
 avg_weights = weights.mean(dim=1).squeeze().numpy()

 print("\n Attention Weights (averaged across heads):")
 print(f"{'':8s}", end="")
 for t in tokens:
 print(f"{t:8s}", end="")
 print()
 for i, t in enumerate(tokens):
 print(f"{t:8s}", end="")
 for j in range(len(tokens)):
 print(f"{avg_weights[i, j]:8.3f}", end="")
 print()

 # ---------------------------------------------------------
 # 2. Train MiniGPT on Character-Level Task
 # ---------------------------------------------------------
 print("\n" + "=" * 50)
 print("PART 2: Train MiniGPT (Character-Level)")
 print("=" * 50)

 text = "attention is all you need transformers have changed everything"
 chars = sorted(list(set(text)))
 vocab_size = len(chars)
 stoi = {ch: i for i, ch in enumerate(chars)}
 itos = {i: ch for ch, i in stoi.items()}

 block_size = 8
 data = torch.tensor([stoi[c] for c in text])

 # Create training pairs
 X, Y = [], []
 for i in range(len(data) - block_size):
 X.append(data[i:i+block_size])
 Y.append(data[i+1:i+block_size+1])
 X, Y = torch.stack(X), torch.stack(Y)

 print(f"Vocab size: {vocab_size}, Training sequences: {len(X)}")

 # Train
 model = MiniGPT(vocab_size, embed_size=64, num_heads=4, num_layers=2, block_size=block_size)
 optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

 for epoch in range(200):
 logits, loss = model(X, Y)
 optimizer.zero_grad()
 loss.backward()
 optimizer.step()
 if (epoch + 1) % 50 == 0:
 print(f" Epoch {epoch+1}/200 — Loss: {loss.item():.4f}")

 # ---------------------------------------------------------
 # 3. Generate Text
 # ---------------------------------------------------------
 print("\n Generated text (greedy decoding):")
 model.eval()
 context = torch.tensor([[stoi['a']]]) # start with 'a'
 generated = ['a']

 with torch.no_grad():
 for _ in range(40):
 logits, _ = model(context[:, -block_size:])
 next_char = logits[:, -1, :].argmax(dim=-1)
 generated.append(itos[next_char.item()])
 context = torch.cat([context, next_char.unsqueeze(1)], dim=1)

 print(f" {''.join(generated)}")

 # ---------------------------------------------------------
 # 4. Model Size
 # ---------------------------------------------------------
 params = sum(p.numel() for p in model.parameters())
 print(f"\n MiniGPT Parameters: {params:,}")

if __name__ == "__main__":
 run_transformer_lab()
