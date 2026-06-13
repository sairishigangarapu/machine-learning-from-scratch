import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time


class LSTMClassifier(nn.Module):
    """LSTM for sequence classification.

    Uses the last hidden state from the top LSTM layer
    to make a classification decision.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.fc(last_hidden)


class GRUClassifier(nn.Module):
    """GRU for sequence classification.

    GRU has no separate cell state — only hidden state.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, h_n = self.gru(x)
        return self.fc(h_n[-1])


class VanillaRNNClassifier(nn.Module):
    """Vanilla RNN for comparison — demonstrates limited memory."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, h_n = self.rnn(x)
        return self.fc(out[:, -1, :])


def generate_sine_data(n_samples=2000, seq_len=50, low_freq=0.1, high_freq=3.0):
    """Generate sine wave frequency classification data.

    Returns:
        X: tensor of shape (n_samples, seq_len, 1) — sine wave signals
        y: tensor of shape (n_samples,) — labels: 0 for low freq, 1 for high freq
    """
    X, y = [], []
    for _ in range(n_samples):
        freq = np.random.uniform(low_freq, high_freq)
        phase = np.random.uniform(0, 2 * np.pi)
        t = np.linspace(0, 2 * np.pi, seq_len)
        signal = np.sin(freq * t + phase)
        label = 0 if freq < 1.0 else 1 if freq > 2.0 else np.random.randint(0, 2)
        X.append(signal)
        y.append(label)
    return torch.FloatTensor(np.array(X)).unsqueeze(-1), torch.LongTensor(np.array(y))


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Compute accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            output = model(bx)
            correct += (output.argmax(1) == by).sum().item()
            total += by.size(0)
    return correct / total


def run_rnn_lab():
    print("=" * 60)
    print("RNN LAB: LSTM, GRU, and Vanilla RNN Comparison")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Generate Sequence Data (Sine Wave Frequency Classification)
    # ---------------------------------------------------------
    print("\n[1] Generating sine wave data...")
    X, y = generate_sine_data(n_samples=3000, seq_len=50)
    split = int(0.8 * len(X))
    train_ds = TensorDataset(X[:split], y[:split])
    test_ds = TensorDataset(X[split:], y[split:])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)

    print(f"    Training samples: {split}")
    print(f"    Test samples:     {len(X) - split}")
    print(f"    Sequence length:  50")
    print(f"    Input features:   1 (sine wave amplitude)")
    print(f"    Classes:          2 (low vs high frequency)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Device:           {device}")

    # ---------------------------------------------------------
    # 2. Train LSTM
    # ---------------------------------------------------------
    print("\n[2] Training LSTM...")
    lstm = LSTMClassifier(input_size=1, hidden_size=64, num_layers=2,
                          num_classes=2, dropout=0.2).to(device)
    optimizer = optim.Adam(lstm.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    lstm_params = count_parameters(lstm)
    print(f"    LSTM parameters: {lstm_params:,}")

    start = time.time()
    for epoch in range(5):
        train_loss = train_epoch(lstm, train_loader, criterion, optimizer, device)
        train_acc = evaluate(lstm, train_loader, device)
        test_acc = evaluate(lstm, test_loader, device)
        print(f"    Epoch {epoch+1}/5 | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    lstm_time = time.time() - start
    lstm_test_acc = evaluate(lstm, test_loader, device)

    # ---------------------------------------------------------
    # 3. Train GRU
    # ---------------------------------------------------------
    print("\n[3] Training GRU...")
    gru = GRUClassifier(input_size=1, hidden_size=64, num_layers=2,
                        num_classes=2, dropout=0.2).to(device)
    optimizer = optim.Adam(gru.parameters(), lr=1e-3)
    gru_params = count_parameters(gru)
    print(f"    GRU parameters: {gru_params:,}")

    start = time.time()
    for epoch in range(5):
        train_loss = train_epoch(gru, train_loader, criterion, optimizer, device)
        train_acc = evaluate(gru, train_loader, device)
        test_acc = evaluate(gru, test_loader, device)
        print(f"    Epoch {epoch+1}/5 | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    gru_time = time.time() - start
    gru_test_acc = evaluate(gru, test_loader, device)

    # ---------------------------------------------------------
    # 4. Train Vanilla RNN (for comparison)
    # ---------------------------------------------------------
    print("\n[4] Training Vanilla RNN...")
    rnn = VanillaRNNClassifier(input_size=1, hidden_size=64, num_layers=2,
                               num_classes=2).to(device)
    optimizer = optim.Adam(rnn.parameters(), lr=1e-3)
    rnn_params = count_parameters(rnn)
    print(f"    Vanilla RNN parameters: {rnn_params:,}")

    start = time.time()
    for epoch in range(5):
        train_loss = train_epoch(rnn, train_loader, criterion, optimizer, device)
        train_acc = evaluate(rnn, train_loader, device)
        test_acc = evaluate(rnn, test_loader, device)
        print(f"    Epoch {epoch+1}/5 | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    rnn_time = time.time() - start
    rnn_test_acc = evaluate(rnn, test_loader, device)

    # ---------------------------------------------------------
    # 5. Comparison
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f" {'Model':20s} | {'Params':>10s} | {'Test Acc':>10s} | {'Time':>8s}")
    print("-" * 55)
    print(f" {'LSTM':20s} | {lstm_params:>10,} | {lstm_test_acc:>10.4f} | {lstm_time:>7.1f}s")
    print(f" {'GRU':20s} | {gru_params:>10,} | {gru_test_acc:>10.4f} | {gru_time:>7.1f}s")
    print(f" {'Vanilla RNN':20s} | {rnn_params:>10,} | {rnn_test_acc:>10.4f} | {rnn_time:>7.1f}s")
    print("-" * 55)

    print("\n    Key observations:")
    print(f"    - LSTM and GRU should outperform Vanilla RNN on this 50-step sequence task")
    print(f"    - GRU has {(gru_params / lstm_params - 1) * 100:.1f}% fewer parameters than LSTM")
    print(f"    - Vanilla RNN has the fewest parameters but worst long-range memory")

    return {
        "lstm_acc": lstm_test_acc,
        "gru_acc": gru_test_acc,
        "rnn_acc": rnn_test_acc,
        "lstm_params": lstm_params,
        "gru_params": gru_params,
        "rnn_params": rnn_params,
    }


if __name__ == "__main__":
    run_rnn_lab()
