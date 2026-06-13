import numpy as np

# ============================================================
# 1. SoftMax from Scratch
# ============================================================

def softmax(z):
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def softmax_stable(z):
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def softmax_jacobian(s):
    m = len(s)
    J = -np.outer(s, s)
    np.fill_diagonal(J, s * (1 - s))
    return J


# ============================================================
# 2. Cross-Entropy from Scratch
# ============================================================

def cross_entropy_single(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-15))

def cross_entropy_batch(y_true, y_pred):
    N = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-15)) / N


# ============================================================
# 3. Combined CE + SoftMax Gradient
# ============================================================

def ce_softmax_gradient(y_true, y_pred):
    return y_pred - y_true


# ============================================================
# 4. Demo: Simple 3-Class Problem
# ============================================================

def demo_3class():
    print("=" * 60)
    print("DEMO: SoftMax + Cross-Entropy on 3-Class Problem")
    print("=" * 60)

    y_true = np.array([0, 1, 0])

    logits_1 = np.array([1.5, 0.5, -0.5])
    logits_2 = np.array([3.0, 1.0, 0.1])
    logits_3 = np.array([0.1, 0.1, 0.1])

    print(f"\nTrue label: {y_true} (class 2)")
    print(f"{'Logits':>30} | {'Softmax':>30} | {'Loss':>8} | {'Gradient':>30}")
    print("-" * 105)

    for z in [logits_1, logits_2, logits_3]:
        y_hat = softmax(z)
        loss = cross_entropy_single(y_true, y_hat)
        grad = ce_softmax_gradient(y_true, y_hat)
        z_str = np.array2string(z, precision=3, separator=", ")
        yh_str = np.array2string(y_hat, precision=4, separator=", ")
        g_str = np.array2string(grad, precision=4, separator=", ")
        print(f"{z_str:>30} | {yh_str:>30} | {loss:>8.4f} | {g_str:>30}")


# ============================================================
# 5. Full Network: 2-Layer MLP with Softmax + Cross-Entropy
# ============================================================

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

class SimpleMLP:
    def __init__(self, input_dim, hidden_dim, output_dim, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.h = relu(self.z1)
        self.z2 = self.h @ self.W2 + self.b2
        self.y_hat = softmax(self.z2)
        return self.y_hat

    def backward(self, x, y, y_hat):
        N = x.shape[0]

        dz2 = (y_hat - y) / N
        dW2 = self.h.T @ dz2
        db2 = np.sum(dz2, axis=0)

        dh = dz2 @ self.W2.T
        dz1 = dh * relu_derivative(self.z1)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return dW1, db1, dW2, db2

    def update(self, dW1, db1, dW2, db2, lr):
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def compute_loss(self, x, y):
        y_hat = self.forward(x)
        return cross_entropy_batch(y, y_hat)

    def accuracy(self, x, y):
        y_hat = self.forward(x)
        return np.mean(np.argmax(y_hat, axis=1) == np.argmax(y, axis=1))


def demo_mlp_training():
    print("\n" + "=" * 60)
    print("DEMO: Full MLP Training with CE + Softmax")
    print("=" * 60)

    np.random.seed(42)
    N = 300
    x = np.random.randn(N, 2)
    y_raw = (x[:, 0]**2 + x[:, 1]**2 > 1.5).astype(int) + \
            ((x[:, 0] > 0) & (x[:, 1] < 0)).astype(int)
    y_raw = np.clip(y_raw, 0, 2)
    y = one_hot(y_raw, 3)

    split = int(0.8 * N)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    mlp = SimpleMLP(input_dim=2, hidden_dim=10, output_dim=3, seed=42)

    print(f"Train samples: {len(x_train)}, Test samples: {len(x_test)}")
    print(f"Input dim: 2, Hidden dim: 10, Output classes: 3")
    print()

    lr = 0.5
    for epoch in range(100):
        y_hat = mlp.forward(x_train)
        dW1, db1, dW2, db2 = mlp.backward(x_train, y_train, y_hat)
        mlp.update(dW1, db1, dW2, db2, lr)

        if epoch % 10 == 0:
            loss = mlp.compute_loss(x_train, y_train)
            acc = mlp.accuracy(x_train, y_train)
            test_acc = mlp.accuracy(x_test, y_test)
            print(f"Epoch {epoch:3d}: Loss={loss:.4f}, "
                  f"Train Acc={acc:.2%}, Test Acc={test_acc:.2%}")

    final_train_acc = mlp.accuracy(x_train, y_train)
    final_test_acc = mlp.accuracy(x_test, y_test)
    print(f"\nFinal: Train Acc={final_train_acc:.2%}, "
          f"Test Acc={final_test_acc:.2%}")


# ============================================================
# 6. Gradient Verification: Numerical vs Analytical
# ============================================================

def numerical_gradient(mlp, x, y, eps=1e-5):
    dW1_num = np.zeros_like(mlp.W1)
    db1_num = np.zeros_like(mlp.b1)
    dW2_num = np.zeros_like(mlp.W2)
    db2_num = np.zeros_like(mlp.b2)

    for i in range(mlp.W1.shape[0]):
        for j in range(mlp.W1.shape[1]):
            orig = mlp.W1[i, j]
            mlp.W1[i, j] = orig + eps
            loss_plus = mlp.compute_loss(x, y)
            mlp.W1[i, j] = orig - eps
            loss_minus = mlp.compute_loss(x, y)
            dW1_num[i, j] = (loss_plus - loss_minus) / (2 * eps)
            mlp.W1[i, j] = orig

    for i in range(mlp.b1.shape[0]):
        orig = mlp.b1[i]
        mlp.b1[i] = orig + eps
        loss_plus = mlp.compute_loss(x, y)
        mlp.b1[i] = orig - eps
        loss_minus = mlp.compute_loss(x, y)
        db1_num[i] = (loss_plus - loss_minus) / (2 * eps)
        mlp.b1[i] = orig

    for i in range(mlp.W2.shape[0]):
        for j in range(mlp.W2.shape[1]):
            orig = mlp.W2[i, j]
            mlp.W2[i, j] = orig + eps
            loss_plus = mlp.compute_loss(x, y)
            mlp.W2[i, j] = orig - eps
            loss_minus = mlp.compute_loss(x, y)
            dW2_num[i, j] = (loss_plus - loss_minus) / (2 * eps)
            mlp.W2[i, j] = orig

    for i in range(mlp.b2.shape[0]):
        orig = mlp.b2[i]
        mlp.b2[i] = orig + eps
        loss_plus = mlp.compute_loss(x, y)
        mlp.b2[i] = orig - eps
        loss_minus = mlp.compute_loss(x, y)
        db2_num[i] = (loss_plus - loss_minus) / (2 * eps)
        mlp.b2[i] = orig

    return dW1_num, db1_num, dW2_num, db2_num


def verify_gradients():
    print("\n" + "=" * 60)
    print("Gradient Verification: Numerical vs Analytical")
    print("=" * 60)

    np.random.seed(123)
    x_small = np.random.randn(5, 2)
    y_small = one_hot(np.array([0, 1, 2, 0, 1]), 3)

    mlp = SimpleMLP(input_dim=2, hidden_dim=4, output_dim=3, seed=123)
    y_hat = mlp.forward(x_small)
    dW1, db1, dW2, db2 = mlp.backward(x_small, y_small, y_hat)

    nW1, nb1, nW2, nb2 = numerical_gradient(mlp, x_small, y_small, eps=1e-5)

    print(f"\nW1 max diff: {np.max(np.abs(dW1 - nW1)):.2e}")
    print(f"b1 max diff: {np.max(np.abs(db1 - nb1)):.2e}")
    print(f"W2 max diff: {np.max(np.abs(dW2 - nW2)):.2e}")
    print(f"b2 max diff: {np.max(np.abs(db2 - nb2)):.2e}")

    all_close = all([
        np.allclose(dW1, nW1, atol=1e-4),
        np.allclose(db1, nb1, atol=1e-4),
        np.allclose(dW2, nW2, atol=1e-4),
        np.allclose(db2, nb2, atol=1e-4),
    ])
    print(f"\nGradients match (allclose at 1e-4): {all_close}")
    if all_close:
        print("=> Analytical backpropagation is CORRECT.")


# ============================================================
# 7. SoftMax Properties Demo
# ============================================================

def softmax_properties():
    print("\n" + "=" * 60)
    print("SoftMax Properties Demo")
    print("=" * 60)

    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)

    print(f"\nLogits:  {logits}")
    print(f"Softmax: {probs}")
    print(f"Sum:     {probs.sum():.4f}")
    print(f"ArgMax:  {np.argmax(logits)} (class with highest logit)")

    print("\nTemperature variation:")
    for T in [0.5, 1.0, 2.0, 10.0]:
        probs_T = softmax(logits / T)
        print(f"  T={T:4.1f}: {np.array2string(probs_T, precision=4, separator=', ')}")

    print("\nJacobian matrix:")
    J = softmax_jacobian(probs)
    print(np.array2string(J, precision=4, separator=", "))
    print(f"Row sums: {J.sum(axis=1)} (should all be 0)")


if __name__ == "__main__":
    demo_3class()
    demo_mlp_training()
    verify_gradients()
    softmax_properties()

    print("\n" + "=" * 60)
    print("All demos complete.")
    print("=" * 60)
