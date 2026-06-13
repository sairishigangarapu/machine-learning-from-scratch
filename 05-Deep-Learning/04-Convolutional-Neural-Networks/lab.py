import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification.

    Architecture:
        Conv(1->32, 3x3) -> ReLU -> MaxPool(2x2) ->
        Conv(32->64, 3x3) -> ReLU -> MaxPool(2x2) ->
        Flatten -> FC(64*7*7 -> 128) -> ReLU -> Dropout(0.5) -> FC(128 -> 10)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimpleCNNFunctional(nn.Module):
    """Same architecture using functional API for educational clarity."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [B, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 32, 7, 7]
        x = x.view(x.size(0), -1)              # flatten
        x = self.fc(x)                         # [B, num_classes]
        return x


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
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
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output = model(batch_X)
            correct += (output.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


def run_cnn_lab():
    print("=" * 60)
    print("CNN LAB: Convolutional Neural Network on MNIST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Data Loading (MNIST)
    # ---------------------------------------------------------
    print("\n[1] Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256)

    print(f"    Training samples: {len(train_data)}")
    print(f"    Test samples:     {len(test_data)}")
    print(f"    Image shape:      {train_data[0][0].shape}")

    # ---------------------------------------------------------
    # 2. Model Setup
    # ---------------------------------------------------------
    print("\n[2] Setting up SimpleCNN model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Device: {device}")

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    params = count_parameters(model)
    print(f"    Trainable parameters: {params:,}")

    # Also show the functional variant's parameter count
    model_fn = SimpleCNNFunctional().to(device)
    params_fn = count_parameters(model_fn)
    print(f"    Functional variant params: {params_fn:,}")

    # ---------------------------------------------------------
    # 3. Training Loop with Epoch-by-Epoch Accuracy
    # ---------------------------------------------------------
    print("\n[3] Training SimpleCNN...")
    epochs = 5
    start_time = time.time()

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)

        elapsed = time.time() - start_time
        print(f"    Epoch {epoch+1:2d}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

    # ---------------------------------------------------------
    # 4. Final Test Evaluation
    # ---------------------------------------------------------
    print("\n[4] Final Evaluation...")
    final_acc = evaluate(model, test_loader, device)
    print(f"    Final Test Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"    Total training time: {total_time:.1f}s")

    # ---------------------------------------------------------
    # 5. Parameter Breakdown
    # ---------------------------------------------------------
    print("\n[5] Parameter Breakdown:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"    {name:40s} shape: {str(list(param.shape)):20s} params: {param.numel():,}")

    print(f"\n{'='*60}")
    print(f"CNN Lab Complete — Final Test Accuracy: {final_acc:.4f}")
    print(f"{'='*60}")

    return final_acc


if __name__ == "__main__":
    run_cnn_lab()
