import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def forward_pass(x, w1, b1, w2, b2, w3, b3, w4):
    z1 = w1 * x + b1
    h1 = sigmoid(z1)
    z2 = w2 * x + b2
    h2 = sigmoid(z2)
    z3 = w3 * h1 + w4 * h2 + b3
    y_pred = sigmoid(z3)
    cache = (z1, h1, z2, h2, z3)
    return y_pred, cache

def backward_pass(x, y, y_pred, cache, w3, w4):
    z1, h1, z2, h2, z3 = cache

    delta_3 = (y_pred - y) * sigmoid_derivative(y_pred)
    dL_dw3 = delta_3 * h1
    dL_dw4 = delta_3 * h2
    dL_db3 = delta_3

    delta_1 = delta_3 * w3 * sigmoid_derivative(h1)
    dL_dw1 = delta_1 * x
    dL_db1 = delta_1

    delta_2 = delta_3 * w4 * sigmoid_derivative(h2)
    dL_dw2 = delta_2 * x
    dL_db2 = delta_2

    return (dL_dw1, dL_db1, dL_dw2, dL_db2, dL_dw3, dL_db3, dL_dw4)

def update_params(params, grads, alpha):
    w1, b1, w2, b2, w3, b3, w4 = params
    dL_dw1, dL_db1, dL_dw2, dL_db2, dL_dw3, dL_db3, dL_dw4 = grads
    w1 -= alpha * dL_dw1; b1 -= alpha * dL_db1
    w2 -= alpha * dL_dw2; b2 -= alpha * dL_db2
    w3 -= alpha * dL_dw3; b3 -= alpha * dL_db3
    w4 -= alpha * dL_dw4
    return (w1, b1, w2, b2, w3, b3, w4)

def compute_loss(y, y_pred):
    return 0.5 * (y - y_pred) ** 2

def train_one_step(x, y, params, alpha=0.1):
    y_pred, cache = forward_pass(x, *params)
    loss = compute_loss(y, y_pred)
    grads = backward_pass(x, y, y_pred, cache, params[4], params[6])
    params = update_params(params, grads, alpha)
    return params, loss, y_pred

def train_network(x_data, y_data, params, alpha=0.1, epochs=200, verbose=True):
    for epoch in range(epochs):
        total_loss = 0.0
        for xi, yi in zip(x_data, y_data):
            params, loss, y_pred = train_one_step(xi, yi, params, alpha)
            total_loss += loss
        avg_loss = total_loss / len(x_data)
        if verbose and epoch % 40 == 0:
            print(f"Epoch {epoch:3d}: Avg Loss = {avg_loss:.6f}, "
                  f"Last Prediction = {y_pred:.4f}")
    return params

def predict(x_data, params):
    predictions = []
    for xi in x_data:
        y_pred, _ = forward_pass(xi, *params)
        predictions.append(y_pred)
    return np.array(predictions)


if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("PART 1: Single Point Training Demo")
    print("=" * 60)

    x_single, y_single = 0.5, 1.0
    params = (1.70, -0.85, -1.70, 0.85, 1.00, 0.00, -1.00)

    print(f"Input: x={x_single}, y={y_single}")
    print(f"Initial params: w1={params[0]:.2f}, b1={params[1]:.2f}, "
          f"w2={params[2]:.2f}, b2={params[3]:.2f}")
    print(f"                w3={params[4]:.2f}, b3={params[5]:.2f}, "
          f"w4={params[6]:.2f}")
    print()

    y_pred, cache = forward_pass(x_single, *params)
    loss = compute_loss(y_single, y_pred)
    print(f"Before training: Pred={y_pred:.4f}, Loss={loss:.6f}")
    print()

    print("Training on single point:")
    for epoch in range(10):
        params, loss, y_pred = train_one_step(x_single, y_single, params, alpha=0.5)
        print(f"  Epoch {epoch+1:2d}: Loss={loss:.6f}, Prediction={y_pred:.4f}")

    print()
    print("=" * 60)
    print("PART 2: Batch Training on Synthetic Data")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 20
    x_train = np.sort(np.random.uniform(0, 1, n_samples))
    y_train = 0.5 + 0.4 * np.sin(x_train * 4 * np.pi) + np.random.normal(0, 0.05, n_samples)

    params = (2.0, -0.5, -2.0, 0.5, 1.5, 0.0, -1.5)

    print(f"Training on {n_samples} samples for 200 epochs...")
    print()

    params = train_network(x_train, y_train, params, alpha=0.2, epochs=200, verbose=True)

    print()
    predictions = predict(x_train, params)
    final_loss = np.mean(0.5 * (y_train - predictions) ** 2)
    print(f"Final Avg Loss: {final_loss:.6f}")
    print(f"Sample predictions vs targets:")
    for i in range(0, n_samples, 4):
        print(f"  x={x_train[i]:.2f}, target={y_train[i]:.4f}, "
              f"pred={predictions[i]:.4f}, error={abs(y_train[i] - predictions[i]):.4f}")

    print()
    print("=" * 60)
    print("PART 3: Learning Rate Comparison")
    print("=" * 60)

    for lr in [0.01, 0.1, 0.5, 1.0]:
        params_lr = (1.70, -0.85, -1.70, 0.85, 1.00, 0.00, -1.00)
        x_demo = np.array([0.2, 0.5, 0.8])
        y_demo = np.array([0.2, 0.7, 0.9])
        params_lr = train_network(x_demo, y_demo, params_lr, alpha=lr,
                                  epochs=100, verbose=False)
        preds = predict(x_demo, params_lr)
        mse = np.mean((y_demo - preds) ** 2)
        print(f"  alpha={lr:.2f}: Final MSE={mse:.6f}, "
              f"Predictions={[f'{p:.3f}' for p in preds]}")
