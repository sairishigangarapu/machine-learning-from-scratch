import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_func(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1.0, alpha * np.exp(x))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gelu_derivative(x):
    c = np.sqrt(2 / np.pi)
    inner = x + 0.044715 * x**3
    tanh_inner = np.tanh(c * inner)
    sech2 = 1 - tanh_inner**2
    d_inner = 1 + 0.134145 * x**2
    return 0.5 * (1 + tanh_inner) + 0.5 * x * c * sech2 * d_inner


def plot_activations():
    x = np.linspace(-5, 5, 1000)

    activations = [
        ("Sigmoid", sigmoid, sigmoid_derivative, "C0"),
        ("Tanh", tanh_func, tanh_derivative, "C1"),
        ("ReLU", relu, relu_derivative, "C2"),
        ("Leaky ReLU", leaky_relu, leaky_relu_derivative, "C3"),
        ("ELU", elu, elu_derivative, "C4"),
        ("GELU", gelu, gelu_derivative, "C5"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, func, deriv, color) in enumerate(activations):
        ax = axes[idx]
        y = func(x)
        dy = deriv(x)

        ax.plot(x, y, label=name, color=color, linewidth=2.5)
        ax.plot(x, dy, label="Derivative", color=color, linestyle="--",
                linewidth=2.0, alpha=0.7)
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-1.5, 5)
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Activation Functions and Their Derivatives",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("activation_functions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[SAVED] activation_functions.png")


def plot_comparison_subplots():
    x = np.linspace(-4, 4, 1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for name, func, color in [
        ("Sigmoid", sigmoid, "C0"),
        ("Tanh", tanh_func, "C1"),
        ("ReLU", relu, "C2"),
        ("Leaky ReLU", leaky_relu, "C3"),
    ]:
        ax1.plot(x, func(x), label=name, color=color, linewidth=2)
        ax2.plot(x, relu_derivative(x) if name == "ReLU"
                 else (leaky_relu_derivative(x) if name == "Leaky ReLU"
                       else sigmoid_derivative(x) if name == "Sigmoid"
                       else tanh_derivative(x)),
                 label=name, color=color, linewidth=2)

    ax1.set_title("Activation Functions", fontsize=14, fontweight="bold")
    ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax1.axvline(x=0, color="gray", linestyle=":", alpha=0.3)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-1.2, 4.2)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Derivatives", fontsize=14, fontweight="bold")
    ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax2.axvline(x=0, color="gray", linestyle=":", alpha=0.3)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-0.2, 1.2)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Activation Comparison: Function vs Derivative",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("activation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[SAVED] activation_comparison.png")


def demonstrate_dying_relu():
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Dying ReLU Effect")
    print("=" * 60)

    np.random.seed(42)
    x = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])

    w, b = 1.0, -1.5
    z = w * x + b
    h_relu = relu(z)
    h_lrelu = leaky_relu(z)
    h_elu = elu(z)

    print(f"Weight w={w:.1f}, Bias b={b:.1f}")
    print(f"Pre-activations z = {w:.1f} * x + ({b:.1f})")
    print()

    print(f"{'x':>6} | {'z':>6} | {'ReLU':>6} | {'LReLU':>7} | {'ELU':>7}")
    print("-" * 45)
    for xi, zi, hr, hl, he in zip(x, z, h_relu, h_lrelu, h_elu):
        print(f"{xi:>6.1f} | {zi:>6.2f} | {hr:>6.2f} | {hl:>7.4f} | {he:>7.4f}")

    dead_fraction = np.mean(z <= 0)
    print(f"\nFraction of dead ReLU neurons: {dead_fraction:.0%}")
    print(f"Leaky ReLU gradients (x<0): {leaky_relu_derivative(z)[z < 0].mean():.4f} (nonzero)")
    print(f"ReLU gradients (x<0): {relu_derivative(z)[z < 0].mean():.0f} (zero)")
    print("=> Dead ReLU neurons stop learning entirely.")
    print("=> Leaky ReLU / ELU keep a small gradient flowing.")


def compare_gradients():
    print("\n" + "=" * 60)
    print("Comparison: Gradient Values Across Activations")
    print("=" * 60)

    x_vals = np.array([-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0])

    print(f"\n{'x':>6} | {'Sigmoid\\' :>9} | {'Tanh\\' :>7} | {'ReLU\\' :>7} | {'LReLU\\' :>8} | {'ELU\\' :>7}")
    print("-" * 60)
    for xv in x_vals:
        sd = sigmoid_derivative(xv)
        td = tanh_derivative(xv)
        rd = relu_derivative(xv)
        ld = leaky_relu_derivative(xv)
        ed = elu_derivative(xv)
        print(f"{xv:>6.1f} | {sd:>9.4f} | {td:>7.4f} | {rd:>7.1f} | {ld:>8.4f} | {ed:>7.4f}")

    print()
    print("Key observation: Sigmoid and Tanh gradients vanish at extremes.")
    print("ReLU gradient is 1 for active neurons (x > 0), preserving signal.")
    print("Leaky ReLU and ELU maintain small nonzero gradients for x < 0.")


def visualize_saturation_effect():
    x = np.linspace(-10, 10, 500)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, name, func, deriv, color in [
        (axes[0], "Sigmoid (saturating)", sigmoid, sigmoid_derivative, "C0"),
        (axes[1], "ReLU (non-saturating)", relu, relu_derivative, "C2"),
    ]:
        y = func(x)
        dy = deriv(x)
        ax.plot(x, y, label=name, color=color, linewidth=2.5)
        ax.fill_between(x, 0, dy, alpha=0.3, color=color,
                        label="Gradient magnitude")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.3)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-0.5, 2.0)
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Saturation Effect: Sigmoid vs ReLU Gradients",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("saturation_effect.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[SAVED] saturation_effect.png")


def relu_network_forward(x, w1, b1, w2, b2, w3, b3, w4):
    z1 = w1 * x + b1
    h1 = relu(z1)
    z2 = w2 * x + b2
    h2 = relu(z2)
    z3 = w3 * h1 + w4 * h2 + b3
    y_pred = sigmoid(z3)
    return y_pred, h1, h2


def sigmoid_network_forward(x, w1, b1, w2, b2, w3, b3, w4):
    z1 = w1 * x + b1
    h1 = sigmoid(z1)
    z2 = w2 * x + b2
    h2 = sigmoid(z2)
    z3 = w3 * h1 + w4 * h2 + b3
    y_pred = sigmoid(z3)
    return y_pred, h1, h2


def compare_network_shapes():
    print("\n" + "=" * 60)
    print("Comparison: ReLU vs Sigmoid Hidden Networks")
    print("=" * 60)

    x_vals = np.linspace(0, 1, 100)
    w1, b1 = 8.0, -4.0
    w2, b2 = -8.0, 4.0
    w3, b3, w4 = 1.0, 0.0, -1.0

    sig_preds = [sigmoid_network_forward(x, w1, b1, w2, b2, w3, b3, w4)[0]
                 for x in x_vals]
    relu_preds = [relu_network_forward(x, w1, b1, w2, b2, w3, b3, w4)[0]
                  for x in x_vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_vals, sig_preds, label="Sigmoid hidden (smooth S-curves)",
            color="C0", linewidth=2)
    ax.plot(x_vals, relu_preds, label="ReLU hidden (piecewise linear)",
            color="C2", linewidth=2, linestyle="--")
    ax.set_xlabel("Input x", fontsize=12)
    ax.set_ylabel("Prediction", fontsize=12)
    ax.set_title("Network Output: Sigmoid vs ReLU Hidden Layers",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("relu_vs_sigmoid_network.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[SAVED] relu_vs_sigmoid_network.png")


if __name__ == "__main__":
    plot_activations()
    plot_comparison_subplots()
    compare_gradients()
    demonstrate_dying_relu()
    visualize_saturation_effect()
    compare_network_shapes()

    print("\n" + "=" * 60)
    print("All plots saved. Open .png files to view.")
    print("=" * 60)
