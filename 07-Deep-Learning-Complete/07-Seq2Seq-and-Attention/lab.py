import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell, outputs  # return all encoder outputs for attention


class AttentionDecoder(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2, output_vocab_size)
        self.W_a = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, hidden, cell, encoder_outputs):
        embedded = self.embedding(x)

        query = hidden[-1:].transpose(0, 1)
        query_proj = self.W_a(query)
        scores = torch.bmm(query_proj, encoder_outputs.transpose(1, 2))
        scores = scores / math.sqrt(encoder_outputs.size(-1))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, encoder_outputs)

        lstm_input = torch.cat([embedded, context], dim=-1)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        combined = torch.cat([output.squeeze(1), context.squeeze(1)], dim=-1)
        prediction = self.fc_out(combined)
        return prediction, hidden, cell, attn_weights


class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        all_attention = torch.zeros(batch_size, trg_len, src.shape[1]).to(self.device)

        hidden, cell, encoder_outputs = self.encoder(src)
        input_token = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            output, hidden, cell, attn = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            all_attention[:, t, :] = attn.squeeze(1)
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs, all_attention


def generate_translation(model, src, max_len=15):
    model.eval()
    with torch.no_grad():
        hidden, cell, encoder_outputs = model.encoder(src)
        input_token = torch.tensor([[1]], device=model.device)  # SOS token index
        generated = [1]
        attention_weights = []

        for _ in range(max_len):
            output, hidden, cell, attn = model.decoder(
                input_token, hidden, cell, encoder_outputs
            )
            attention_weights.append(attn.squeeze(0).squeeze(0).cpu().numpy())
            top1 = output.argmax(1).item()
            generated.append(top1)
            if top1 == 2:  # EOS token index
                break
            input_token = torch.tensor([[top1]], device=model.device)

    return generated, np.array(attention_weights)


def visualize_attention(attention_matrix, source_tokens, target_tokens, save_path="attention_heatmap.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attention_matrix, cmap='Blues')

    ax.set_xticks(np.arange(len(source_tokens)))
    ax.set_yticks(np.arange(len(target_tokens)))
    ax.set_xticklabels(source_tokens)
    ax.set_yticklabels(target_tokens)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(attention_matrix.shape[0]):
        for j in range(attention_matrix.shape[1]):
            ax.text(j, i, f"{attention_matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)

    ax.set_xlabel("Source tokens")
    ax.set_ylabel("Target tokens")
    ax.set_title("Attention Weights Heatmap")

    fig.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"Attention heatmap saved to {save_path}")


def print_attention_grid(attn_matrix, source_words, target_words):
    print("\nAttention Weights (target rows x source columns):")
    header = " " * 8 + " ".join(f"{s:>10}" for s in source_words)
    print(header)
    for i, t_word in enumerate(target_words):
        row = f"{t_word:>8}"
        for j in range(len(source_words)):
            row += f"{attn_matrix[i, j]:>10.3f}"
        print(row)


def main():
    print("=" * 65)
    print("Lab: Seq2Seq with Attention — Forward Pass & Attention Visualization")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -----------------------------------------------------------------
    # 1. Scaled Dot-Product Attention (NumPy)
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 1: Scaled Dot-Product Attention from Scratch (NumPy)")
    print("-" * 65)

    np.random.seed(42)
    batch_size, n_queries, n_keys, d_k = 2, 3, 4, 8
    Q = np.random.randn(batch_size, n_queries, d_k)
    K = np.random.randn(batch_size, n_keys, d_k)
    V = np.random.randn(batch_size, n_keys, d_k)

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn_np = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    output_np = attn_np @ V

    print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
    print(f"Attention weights shape: {attn_np.shape}")
    print(f"Output shape: {output_np.shape}")
    print(f"Attention weights sum to 1 (per query): {attn_np.sum(axis=-1)}")

    # -----------------------------------------------------------------
    # 2. Demo Attention Matrix Visualization
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 2: Attention Matrix Visualization (Sample Translation)")
    print("-" * 65)

    source_words = ["don't", "eat", "the", "pizza"]
    target_words = ["no", "comas", "la", "pizza"]

    attn_example = np.array([
        [0.85, 0.10, 0.03, 0.02],
        [0.05, 0.80, 0.10, 0.05],
        [0.02, 0.05, 0.88, 0.05],
        [0.02, 0.03, 0.05, 0.90],
    ])

    print_attention_grid(attn_example, source_words, target_words)
    print("\nInterpretation:")
    print("  - 'no' attends strongly to \"don't\" (0.85) — carries negation")
    print("  - 'comas' attends to 'eat' (0.80) — verb translation")
    print("  - 'la' attends to 'the' (0.88) — article translation")
    print("  - 'pizza' attends to 'pizza' (0.90) — noun translation")

    # -----------------------------------------------------------------
    # 3. Build Seq2Seq with Attention in PyTorch
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 3: Seq2Seq with Attention — Model Forward Pass")
    print("-" * 65)

    torch.manual_seed(42)
    vocab_size = 20
    embed_dim = 16
    hidden_dim = 32

    enc = Encoder(vocab_size, embed_dim, hidden_dim)
    dec = AttentionDecoder(vocab_size, embed_dim, hidden_dim)
    model = Seq2SeqAttention(enc, dec, device).to(device)

    src = torch.randint(3, vocab_size, (4, 8)).to(device)
    trg = torch.randint(3, vocab_size, (4, 10)).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {total_params:,}")
    print(f"Encoder input: {src.shape}")
    print(f"Decoder target: {trg.shape}")

    outputs, attention = model(src, trg, teacher_forcing_ratio=0.8)
    print(f"Output shape: {outputs.shape}")
    print(f"Attention shape: {attention.shape}")

    # Verify attention weights sum to 1 per target position
    attn_sum = attention.sum(dim=-1)
    print(f"Attention sums to 1 (per sample 0): {attn_sum[0].detach().cpu().numpy()}")

    # -----------------------------------------------------------------
    # 4. Generate Translation with Trained Model
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 4: Greedy Generation (Inference Mode)")
    print("-" * 65)

    with torch.no_grad():
        gen_tokens, gen_attn = generate_translation(
            model, src[:1], max_len=12
        )

    print(f"Generated token sequence: {gen_tokens}")
    print(f"Attention matrix shape: {gen_attn.shape}")

    # -----------------------------------------------------------------
    # 5. Visualize Attention Weights
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 5: Attention Weights Heatmap")
    print("-" * 65)

    src_labels = [f"s{i}" for i in range(src.shape[1])]
    trg_labels = [f"t{i}" for i in range(gen_attn.shape[0])]

    print("\nAttention weights from generated sequence:")
    print_attention_grid(gen_attn, src_labels, trg_labels)

    visualize_attention(
        gen_attn,
        src_labels,
        trg_labels,
        save_path="attention_heatmap.png"
    )

    # -----------------------------------------------------------------
    # 6. Training a Toy Seq2Seq Model (Quick Demo)
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 6: Training Demo on Synthetic Data")
    print("-" * 65)

    torch.manual_seed(42)
    dataset_size = 100
    src_len = 6
    trg_len = 8

    X_train = torch.randint(3, vocab_size, (dataset_size, src_len)).to(device)
    Y_train = torch.randint(3, vocab_size, (dataset_size, trg_len)).to(device)

    enc_train = Encoder(vocab_size, embed_dim, hidden_dim)
    dec_train = AttentionDecoder(vocab_size, embed_dim, hidden_dim)
    model_train = Seq2SeqAttention(enc_train, dec_train, device).to(device)
    optimizer = torch.optim.Adam(model_train.parameters(), lr=5e-3)

    print(f"Training on {dataset_size} synthetic pairs...")
    for epoch in range(100):
        model_train.train()
        outputs, _ = model_train(X_train, Y_train, teacher_forcing_ratio=0.7)
        loss = F.cross_entropy(
            outputs[:, 1:, :].reshape(-1, vocab_size),
            Y_train[:, 1:].reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_train.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/100 — Loss: {loss.item():.4f}")

    print(f"Training complete. Final loss: {loss.item():.4f}")

    # Generate after training
    with torch.no_grad():
        gen_tokens_trained, gen_attn_trained = generate_translation(
            model_train, X_train[:1], max_len=trg_len
        )
    print(f"Generated sequence after training: {gen_tokens_trained}")

    print("\n" + "=" * 65)
    print("Lab complete! Key takeaways:")
    print("  1. Encoder reads input, produces hidden states for all positions")
    print("  2. Attention lets decoder dynamically attend to relevant input tokens")
    print("  3. Scaled dot-product prevents softmax saturation for large d_k")
    print("  4. Attention weights form an alignment matrix (interpretable)")
    print("  5. Teacher forcing stabilizes training but causes exposure bias")
    print("=" * 65)


if __name__ == "__main__":
    main()
