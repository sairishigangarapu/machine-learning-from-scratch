import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =========================================================================
# Core Transformer Components
# =========================================================================

class SelfAttention(nn.Module):
    """Multi-head self-attention with optional causal masking."""
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert embed_size % num_heads == 0

        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)
        self.W_o = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None, return_weights=False):
        B, T, C = x.shape

        q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        output = self.W_o(out)

        if return_weights:
            return output, attn
        return output


class TransformerBlock(nn.Module):
    """One transformer block: Self-Attention + Add&Norm + FFN + Add&Norm."""
    def __init__(self, embed_size, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = SelfAttention(embed_size, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_size),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_weights=False):
        attn_out = self.attn(self.norm1(x), mask, return_weights)
        if return_weights:
            attn_out, weights = attn_out
        x = x + self.drop(attn_out)
        x = x + self.ff(self.norm2(x))
        if return_weights:
            return x, weights
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (from original Transformer paper)."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MiniGPT(nn.Module):
    """A minimal decoder-only transformer (GPT-like) for character-level text."""
    def __init__(self, vocab_size, embed_size=128, num_heads=4, num_layers=4, block_size=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(block_size, embed_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, embed_size * 4)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size)
        self.block_size = block_size
        self.embed_size = embed_size
        self.num_heads = num_heads

    def _build_causal_mask(self, T, device):
        """Build causal mask: upper-triangular of -inf."""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        mask = ~mask  # True means attend
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    def forward(self, idx, targets=None, return_weights=False):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        all_weights = [] if return_weights else None
        causal_mask = self._build_causal_mask(T, idx.device)

        for block in self.blocks:
            if return_weights:
                x, w = block(x, mask=causal_mask, return_weights=True)
                all_weights.append(w)
            else:
                x = block(x, mask=causal_mask)

        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        if return_weights:
            return logits, loss, all_weights
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    topk_vals, _ = torch.topk(logits, top_k, dim=-1)
                    logits[logits < topk_vals[:, -1:]] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, idx_next], dim=1)
        return idx


# =========================================================================
# Helper functions
# =========================================================================

def compute_attention_weights(model, tokens):
    """Extract and average attention weights from all layers and heads."""
    model.eval()
    with torch.no_grad():
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        # Ensure sequence length does not exceed model's block_size
        tokens = tokens[:, :model.block_size]
        _, _, all_weights = model(tokens, return_weights=True)

    # all_weights: list of (batch, heads, T, T) per layer
    # Average across layers and heads
    stacked = torch.stack(all_weights)  # (L, B, H, T, T)
    avg_across_heads = stacked.mean(dim=2)  # (L, B, T, T)
    avg_across_layers = avg_across_heads.mean(dim=0)  # (B, T, T)
    return avg_across_layers.squeeze(0).cpu().numpy()


def print_attention_matrix(attn_matrix, tokens, title="Attention Weights"):
    print(f"\n{title}:")
    n = len(tokens)
    header = " " * 10 + " ".join(f"{t:>8}" for t in tokens)
    print(header)
    for i in range(n):
        row = f"{tokens[i]:>10}"
        for j in range(n):
            row += f"{attn_matrix[i, j]:>8.3f}"
        print(row)


def plot_attention_heatmap(attn_matrix, tokens, save_path="attention_heatmap.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_matrix, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(attn_matrix.shape[0]):
        for j in range(attn_matrix.shape[1]):
            ax.text(j, i, f"{attn_matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_xlabel("Keys (attended to)")
    ax.set_ylabel("Queries (attending from)")
    ax.set_title("Self-Attention Weights (averaged across heads and layers)")

    fig.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"Heatmap saved to {save_path}")


# =========================================================================
# Main demo
# =========================================================================

def main():
    print("=" * 65)
    print("Lab: Transformers — Self-Attention, MiniGPT, Text Generation")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -----------------------------------------------------------------
    # PART 1: Scaled Dot-Product Attention (Manual Computation)
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 1: Scaled Dot-Product Attention from Scratch")
    print("-" * 65)

    torch.manual_seed(42)
    d_k = 8
    n_queries, n_keys = 3, 4
    Q = torch.randn(1, n_queries, d_k)
    K = torch.randn(1, n_keys, d_k)
    V = torch.randn(1, n_keys, d_k)

    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    attn_manual = F.softmax(scores, dim=-1)
    output_manual = attn_manual @ V

    print(f"Input Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
    print(f"Attention weights shape: {attn_manual.shape}")
    print(f"Output shape: {output_manual.shape}")
    print(f"Attention sums to 1 (per query): {attn_manual.sum(dim=-1).squeeze()}")

    # -----------------------------------------------------------------
    # PART 2: Self-Attention Visualization
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 2: Multi-Head Self-Attention Visualization")
    print("-" * 65)

    torch.manual_seed(42)
    attn_layer = SelfAttention(embed_size=64, num_heads=8)

    # Simulate a short sequence
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    x = torch.randn(1, len(tokens), 64)

    with torch.no_grad():
        q = attn_layer.W_q(x).view(1, -1, 8, 8).transpose(1, 2)
        k = attn_layer.W_k(x).view(1, -1, 8, 8).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (8 ** 0.5)
        weights = F.softmax(scores, dim=-1)

    # Average across heads
    avg_weights = weights.mean(dim=1).squeeze().cpu().numpy()

    print_attention_matrix(avg_weights, tokens, "Self-Attention: raw weights (averaged over 8 heads)")

    # -----------------------------------------------------------------
    # PART 3: Positional Encoding
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 3: Sinusoidal Positional Encoding")
    print("-" * 65)

    pe = PositionalEncoding(d_model=64, max_len=50)
    sample = torch.zeros(1, 10, 64)
    out = pe(sample)
    print(f"Input shape: {sample.shape}, Output shape: {out.shape}")
    print(f"Positional encoding buffer shape: {pe.pe.shape}")
    print("(Positional encoding values are added to token embeddings)")

    # -----------------------------------------------------------------
    # PART 4: Train MiniGPT on Character-Level Text
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 4: Training MiniGPT (Character-Level Language Model)")
    print("-" * 65)

    text = "attention is all you need transformers have changed everything in machine learning"
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    print(f"Unique characters: {''.join(chars)}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Text length: {len(text)} characters")

    block_size = 8
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    # Create training pairs
    X, Y = [], []
    for i in range(len(data) - block_size):
        X.append(data[i:i+block_size])
        Y.append(data[i+1:i+block_size+1])
    X = torch.stack(X).to(device)
    Y = torch.stack(Y).to(device)

    print(f"Number of training sequences: {len(X)}")

    # Initialize model
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_size=64,
        num_heads=4,
        num_layers=2,
        block_size=block_size
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_losses = []
    for epoch in range(250):
        logits, loss = model(X, Y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}/250 — Loss: {loss.item():.4f}")

    print(f"Training complete. Final loss: {loss.item():.4f}")

    # -----------------------------------------------------------------
    # PART 5: Text Generation
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 5: Text Generation with MiniGPT")
    print("-" * 65)

    model.eval()

    # Greedy decoding
    context = torch.tensor([[stoi['a']]], device=device)
    generated_greedy = ['a']

    with torch.no_grad():
        for _ in range(60):
            logits, _ = model(context[:, -block_size:])
            next_char = logits[:, -1, :].argmax(dim=-1).item()
            generated_greedy.append(itos[next_char])
            context = torch.cat([
                context,
                torch.tensor([[next_char]], device=device)
            ], dim=1)

    greedy_text = ''.join(generated_greedy)
    print(f"Greedy generation (deterministic):")
    print(f"  {greedy_text}")

    # Temperature sampling
    context = torch.tensor([[stoi['a']]], device=device)
    generated_sampled = ['a']

    with torch.no_grad():
        for _ in range(60):
            logits, _ = model(context[:, -block_size:])
            logits = logits[:, -1, :] / 0.8
            probs = F.softmax(logits, dim=-1)
            next_char = torch.multinomial(probs, num_samples=1).item()
            generated_sampled.append(itos[next_char])
            context = torch.cat([
                context,
                torch.tensor([[next_char]], device=device)
            ], dim=1)

    sampled_text = ''.join(generated_sampled)
    print(f"Sampled generation (temperature=0.8):")
    print(f"  {sampled_text}")

    # -----------------------------------------------------------------
    # PART 6: Attention Weights Visualization
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 6: Attention Weights from Trained Model")
    print("-" * 65)

    test_tokens_list = ['a', 't', 't', 'e', 'n', 't', 'i']
    test_indices = torch.tensor([[stoi[c] for c in test_tokens_list]], device=device)

    attn_matrix = compute_attention_weights(model, test_indices)

    print_attention_matrix(attn_matrix, test_tokens_list,
                           "Self-Attention (MiniGPT, avg over layers & heads)")

    plot_attention_heatmap(attn_matrix, test_tokens_list,
                           save_path="transformer_attention_heatmap.png")

    # -----------------------------------------------------------------
    # PART 7: Causal Mask Visualization
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 7: Causal Mask — How GPT Prevents Future Peeking")
    print("-" * 65)

    T = 6
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    print(f"Causal mask (6x6), True = -inf (blocked):")
    print(mask.int().numpy())

    mask_attend = ~mask
    print(f"\nAttention allowed (True = can attend):")
    print(mask_attend.int().numpy())

    print("\nToken-to-token visibility:")
    token_labels = ["x1", "x2", "x3", "x4", "x5", "x6"]
    for i in range(T):
        visible = [token_labels[j] for j in range(i+1)]
        print(f"  {token_labels[i]} can see: {', '.join(visible)}")

    # -----------------------------------------------------------------
    # PART 8: Model Size and Summary
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 8: Model Summary")
    print("-" * 65)

    print(f"MiniGPT config:")
    print(f"  Embedding size: 64")
    print(f"  Number of heads: 4")
    print(f"  Number of layers: 2")
    print(f"  Block size: 8")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Total parameters: {total_params:,}")

    params_by_component = {
        "Token embedding": model.token_emb.weight.numel(),
        "Position embedding": model.pos_emb.weight.numel(),
        "Transformer blocks": sum(p.numel() for p in model.blocks.parameters()),
        "Layer norm + head": model.norm.weight.numel() + model.head.weight.numel() + model.head.bias.numel(),
    }
    for name, count in params_by_component.items():
        print(f"  {name}: {count:,} parameters")

    print("\n" + "=" * 65)
    print("Lab complete! Key takeaways:")
    print("  1. Self-attention computes pairwise relevance between all tokens")
    print("  2. Multi-head attention captures diverse relationship types")
    print("  3. Causal masking ensures autoregressive generation")
    print("  4. Positional encoding injects order information")
    print("  5. Decoder-only models (like GPT) generate text via next-token prediction")
    print("=" * 65)


if __name__ == "__main__":
    main()
