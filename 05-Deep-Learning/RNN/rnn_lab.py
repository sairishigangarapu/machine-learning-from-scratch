import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.fc(last_hidden)

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, h_n = self.gru(x)
        return self.fc(h_n[-1])

def generate_sine_data(n_samples=1000, seq_len=50):
    X, y = [], []
    for _ in range(n_samples):
        freq = np.random.uniform(0.1, 3.0)
        phase = np.random.uniform(0, 2 * np.pi)
        t = np.linspace(0, 2 * np.pi, seq_len)
        signal = np.sin(freq * t + phase)
        label = 0 if freq < 1.0 else 1
        X.append(signal)
        y.append(label)
    return torch.FloatTensor(np.array(X)).unsqueeze(-1), torch.LongTensor(y)

def run_rnn_lab():
    # ---------------------------------------------------------
    # 1. Generate Sequence Data
    # ---------------------------------------------------------
    X, y = generate_sine_data(n_samples=2000, seq_len=50)
    split = int(0.8 * len(X))
    train_ds = TensorDataset(X[:split], y[:split])
    test_ds = TensorDataset(X[split:], y[split:])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)

    print(f"📊 Train: {split}, Test: {len(X)-split}, Seq length: 50")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------------------------------------------------
    # 2. Train LSTM
    # ---------------------------------------------------------
    lstm = LSTMClassifier(input_size=1, hidden_size=64, num_layers=2, num_classes=2).to(device)
    optimizer = optim.Adam(lstm.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("\n🔁 Training LSTM...")
    for epoch in range(5):
        lstm.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(lstm(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        lstm.eval()
        correct = sum((lstm(bx.to(device)).argmax(1) == by.to(device)).item()
                       for bx, by in test_loader)
        acc = correct / len(test_ds)
        print(f"  Epoch {epoch+1}/5 — Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.4f}")

    # ---------------------------------------------------------
    # 3. Train GRU
    # ---------------------------------------------------------
    gru = GRUClassifier(input_size=1, hidden_size=64, num_layers=2, num_classes=2).to(device)
    optimizer = optim.Adam(gru.parameters(), lr=1e-3)

    print("\n🔁 Training GRU...")
    for epoch in range(5):
        gru.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(gru(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        gru.eval()
        correct = sum((gru(bx.to(device)).argmax(1) == by.to(device)).item()
                       for bx, by in test_loader)
        acc = correct / len(test_ds)
        print(f"  Epoch {epoch+1}/5 — Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.4f}")

    # ---------------------------------------------------------
    # 4. Compare
    # ---------------------------------------------------------
    lstm_params = sum(p.numel() for p in lstm.parameters())
    gru_params = sum(p.numel() for p in gru.parameters())
    print(f"\n📐 LSTM Parameters: {lstm_params:,}")
    print(f"📐 GRU  Parameters: {gru_params:,}")

if __name__ == "__main__":
    run_rnn_lab()
