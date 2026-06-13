## L30: Coding Neural Networks with PyTorch + Lightning

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. What Lightning Adds

### Motivation and Intuition

PyTorch Lightning is a lightweight wrapper that **structures the training code** and removes boilerplate. It gives you:

| Feature | Vanilla PyTorch | With Lightning |
| :--- | :--- | :--- |
| Device management | Manual `.to(device)` | Automatic |
| Training loop | Write from scratch | Built into `Trainer` |
| Validation loop | Manual | Automatic |
| GPU/TPU acceleration | Manual `.cuda()` | `Trainer(accelerator="gpu")` |
| Mixed precision | Manual | `Trainer(precision=16)` |
| Logging | Manual | Integrates with TensorBoard, WandB, etc. |
| Checkpointing | Manual | Automatic best-model saving |
| Learning rate finder | Not built-in | `trainer.tune(model)` |
| Multi-GPU | Manual DDP setup | `Trainer(devices=4)` |

```python
import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
```

---

## 2. LightningModule

### Motivation and Intuition

Instead of a raw `nn.Module`, you subclass `LightningModule`. This organizes your code into standard hooks that the `Trainer` calls automatically.

```python
class MyLightningModel(L.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.01)
```

| Method | Purpose | Called by Trainer |
| :--- | :--- | :--- |
| `training_step` | One forward + backward pass on a batch | Every batch |
| `validation_step` | Validation loop logic | Every `val_check_interval` |
| `configure_optimizers` | Return optimizer(s) and scheduler(s) | Once at start |
| `forward` | Inference | User calls `model(x)` |
| `test_step` | Test loop logic | `trainer.test()` |
| `on_epoch_end` | Hook after each epoch | Each epoch end |

---

## 3. Trainer

### Motivation and Intuition

The `Trainer` automates the entire training loop. You just define the model and data, and the trainer handles the rest.

```python
from torch.utils.data import DataLoader, TensorDataset

# Data
X = torch.linspace(-1, 1, 100).reshape(-1, 1)
y = 2 * X + 1 + 0.1 * torch.randn_like(X)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model
model = MyLightningModel(input_size=1, hidden_size=10, output_size=1)

# Trainer
trainer = L.Trainer(
    max_epochs=100,
    accelerator="auto",             # detects GPU if available
    log_every_n_steps=10,
    enable_checkpointing=True,
)

trainer.fit(model, dataloader)
```

### Trainer Key Arguments

| Argument | Effect |
| :--- | :--- |
| `max_epochs` | Number of training epochs |
| `accelerator` | `"auto"`, `"gpu"`, `"cpu"`, `"tpu"` |
| `devices` | Number of GPUs: 1, 4, `"auto"` |
| `precision` | `32`, `16` (mixed precision), `bf16` |
| `log_every_n_steps` | Logging frequency |
| `enable_checkpointing` | Saves best model automatically |
| `val_check_interval` | How often to run validation (epochs or batches) |
| `overfit_batches` | Overfit on small subset for debugging |

---

## 4. Automatic Learning Rate Finding

### Motivation and Intuition

Choosing a learning rate is often trial-and-error. Lightning's learning rate finder runs a small loop: it increases the LR exponentially and tracks the loss. The optimal LR is where the loss descends most steeply.

```python
trainer = L.Trainer(max_epochs=100)
lr_finder = trainer.tune(model, dataloader)
model.hparams.lr = lr_finder.suggestion()
```

Alternatively, the Trainer can auto-tune if `auto_lr_find=True` is set and you use `self.lr` or `self.learning_rate` in `configure_optimizers`.

---

## 5. GPU Acceleration

### Motivation and Intuition

With Lightning, multi-GPU and mixed-precision training require no code changes — just Trainer arguments.

```python
# Single GPU
trainer = L.Trainer(accelerator="gpu", devices=1)

# Multi-GPU (distributed data parallel)
trainer = L.Trainer(accelerator="gpu", devices=4, strategy="ddp")

# Mixed precision (faster, less memory)
trainer = L.Trainer(accelerator="gpu", precision=16)

# TPU
trainer = L.Trainer(accelerator="tpu", devices=8)
```

Lightning handles gradient synchronization, device assignment, and data splitting across GPUs automatically.

---

## 6. Logging and Checkpointing

### Motivation and Intuition

Lightning integrates with TensorBoard, CSV, WandB, and MLflow. Checkpoints save model weights, optimizer state, and epoch — enabling resume training.

```python
from lightning.pytorch.loggers import TensorBoardLogger

logger = TensorBoardLogger("logs", name="my_model")

# Callbacks
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    filename="best-{epoch:02d}",
)

early_stop = EarlyStopping(monitor="val_loss", patience=10)

trainer = L.Trainer(
    max_epochs=100,
    logger=logger,
    callbacks=[checkpoint_callback, early_stop],
)

trainer.fit(model, dataloader)

# Load best checkpoint
best_model = MyLightningModel.load_from_checkpoint(checkpoint_callback.best_model_path)
```

---

## 7. Complete Example

```python
import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Data
X = torch.linspace(-1, 1, 1000).reshape(-1, 1)
y = 2 * X + 1 + 0.1 * torch.randn_like(X)
dataset = TensorDataset(X, y)
train_dl = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
class LinearModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(1, 1)
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.01)

model = LinearModel()

# Training
logger = L.pytorch.loggers.CSVLogger("logs")
trainer = L.Trainer(max_epochs=50, logger=logger, accelerator="auto")
trainer.fit(model, train_dl)
```

---

> **Check your intuition:** What is the benefit of separating `training_step` from the training loop? How does this design make multi-GPU training transparent?

---

## Prerequisites and Further Reading

- **StatQuest:** Introduction to PyTorch (L29), Tensors for Neural Networks (L26)
- **Lightning docs:** LightningModule, Trainer, callbacks, loggers
- **Concepts:** Training loops, backpropagation, GPU parallelism (DDP)
