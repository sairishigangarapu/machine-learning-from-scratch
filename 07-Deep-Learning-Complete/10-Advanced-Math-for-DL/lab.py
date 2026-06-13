"""
10-Advanced-Math-for-DL/lab.py
Tensor and matrix operations for deep learning.
Covers: tensor creation, matmul, broadcasting, transposition,
        attention scores, einsum, gradient flow.
"""

import torch

# ============================================================
# PART 1: Creating Tensors of Various Shapes
# ============================================================
print("=" * 60)
print("PART 1: Tensor Creation and Attributes")
print("=" * 60)

scalar = torch.tensor(3.14159)
vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
tensor_3d = torch.randn(2, 3, 4)
tensor_4d = torch.randn(2, 3, 4, 5)

print(f"Scalar:   shape={scalar.shape},   numel={scalar.numel()},  value={scalar.item()}")
print(f"Vector:   shape={vector.shape},   numel={vector.numel()}")
print(f"Matrix:   shape={matrix.shape},   numel={matrix.numel()}")
print(f"3D-tensor: shape={tensor_3d.shape}, numel={tensor_3d.numel()}")
print(f"4D-tensor: shape={tensor_4d.shape}, numel={tensor_4d.numel()}")

# Strides
print(f"\nMatrix strides: {matrix.stride()}  -- (row_step, col_step)")
print(f"Transposed strides: {matrix.T.stride()}")

# Contiguous check
print(f"\nMatrix is contiguous: {matrix.is_contiguous()}")
print(f"Transposed is contiguous: {matrix.T.is_contiguous()}")

# ============================================================
# PART 2: Matrix Multiplication and Broadcasting
# ============================================================
print("\n" + "=" * 60)
print("PART 2: Matrix Multiplication and Broadcasting")
print("=" * 60)

# Forward pass: Wx + b
B, D, H = 4, 3, 2
X = torch.randn(B, D)
W = torch.randn(D, H)
b = torch.randn(1, H)

Y = X @ W + b  # broadcasting adds b to every row
print(f"X shape: {X.shape}, W shape: {W.shape}, b shape: {b.shape}")
print(f"Y = X @ W + b -> {Y.shape}")

# Broadcasting demo
print("\nBroadcasting examples:")
a = torch.tensor([[1.0], [2.0], [3.0]])      # shape (3, 1)
b_ten = torch.tensor([10.0, 20.0, 30.0])      # shape (3,) -> broadcasts to (1, 3) -> (3, 3)
c = a + b_ten
print(f"  ({a.shape}) + ({b_ten.shape}) -> ({c.shape})")
print(f"  Result:\n{c}")

# Incompatible shapes demo
try:
    a_bad = torch.randn(3, 2)
    b_bad = torch.randn(3,)
    _ = a_bad + b_bad
except RuntimeError as e:
    print(f"  Incompatible (3,2) + (3,): {e}")

# ============================================================
# PART 3: Transposition for Attention
# ============================================================
print("\n" + "=" * 60)
print("PART 3: Transposition and Attention Scores")
print("=" * 60)

N, d_k = 4, 8
Q = torch.randn(N, d_k)
K = torch.randn(N, d_k)
V = torch.randn(N, d_k)

# Without transpose: wrong
try:
    bad = Q @ K
except RuntimeError as e:
    print(f"  Q ({Q.shape}) @ K ({K.shape}) FAILS: {e}")

# With transpose: correct
scores = Q @ K.T / (d_k ** 0.5)       # (N, N)
A = torch.softmax(scores, dim=-1)
output = A @ V                         # (N, d_k)
print(f"  Q @ K.T / sqrt(d_k) -> scores: {scores.shape}")
print(f"  softmax(scores)      -> attn:   {A.shape}")
print(f"  A @ V                -> output: {output.shape}")
print(f"  Row 0 sums to {A[0].sum().item():.4f}")

# Batched transposition
B = 3
Q_batch = torch.randn(B, N, d_k)
K_batch = torch.randn(B, N, d_k)
scores_batch = Q_batch @ K_batch.transpose(-2, -1) / (d_k ** 0.5)
print(f"  Batched scores: {scores_batch.shape}")

# Causal mask
mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
scores_masked = scores_batch.masked_fill(mask.unsqueeze(0), float("-inf"))
A_masked = torch.softmax(scores_masked, dim=-1)
print(f"  Causal masked scores example:")
print(f"    Before mask, first row:\n     {scores_batch[0, 0].detach().round(decimals=2)}")
print(f"    After mask:\n     {scores_masked[0, 0].detach().round(decimals=2)}")

# ============================================================
# PART 4: Einsum for Batched Operations
# ============================================================
print("\n" + "=" * 60)
print("PART 4: Einstein Summation (Einsum)")
print("=" * 60)

# Standard matmul via einsum
A = torch.randn(4, 5)
B_mat = torch.randn(5, 6)
C1 = A @ B_mat
C2 = torch.einsum("ij,jk->ik", A, B_mat)
print(f"  Standard matmul: {C1.shape}")
print(f"  Einsum matmul:   {C2.shape}")
print(f"  Match: {torch.allclose(C1, C2)}")

# Batched matmul via einsum
A_batch = torch.randn(3, 4, 5)
B_batch = torch.randn(3, 5, 6)
C_batch = torch.einsum("bij,bjk->bik", A_batch, B_batch)
print(f"  Batched einsum: {C_batch.shape}")

# Batched multi-head attention scores via einsum
B_b, H, T, d = 2, 8, 16, 64
Q_mh = torch.randn(B_b, H, T, d)
K_mh = torch.randn(B_b, H, T, d)
V_mh = torch.randn(B_b, H, T, d)

scores_ein = torch.einsum("bhtd,bhTd->bhtT", Q_mh, K_mh) / (d ** 0.5)
weights_ein = torch.softmax(scores_ein, dim=-1)
context_ein = torch.einsum("bhtT,bhTd->bhtd", weights_ein, V_mh)
print(f"  Multi-head scores:  {scores_ein.shape}")
print(f"  Multi-head weights: {weights_ein.shape}")
print(f"  Multi-head context: {context_ein.shape}")

# Dot product and outer product
a = torch.tensor([1.0, 2.0, 3.0])
b_vec = torch.tensor([4.0, 5.0, 6.0])
dot = torch.einsum("i,i->", a, b_vec)
outer = torch.einsum("i,j->ij", a, b_vec)
print(f"  Einsum dot product:  {dot.item()}")
print(f"  Einsum outer product:\n{outer}")

# ============================================================
# PART 5: Multi-Head Attention End-to-End
# ============================================================
print("\n" + "=" * 60)
print("PART 5: Multi-Head Attention (Explicit Shapes)")
print("=" * 60)

B, N, d_model, H = 2, 16, 512, 8
d_k = d_model // H

X = torch.randn(B, N, d_model)

# Project to Q, K, V
W_Q = torch.randn(d_model, d_model)
W_K = torch.randn(d_model, d_model)
W_V = torch.randn(d_model, d_model)
W_O = torch.randn(d_model, d_model)

Q_proj = X @ W_Q
K_proj = X @ W_K
V_proj = X @ W_V
print(f"  After projection: Q {Q_proj.shape}, K {K_proj.shape}, V {V_proj.shape}")

# Reshape to (B, H, N, d_k)
Q_h = Q_proj.view(B, N, H, d_k).transpose(1, 2)
K_h = K_proj.view(B, N, H, d_k).transpose(1, 2)
V_h = V_proj.view(B, N, H, d_k).transpose(1, 2)
print(f"  After split: Q {Q_h.shape}, K {K_h.shape}, V {V_h.shape}")

# Scaled dot-product attention
scores_mh = Q_h @ K_h.transpose(-2, -1) / (d_k ** 0.5)   # (B, H, N, N)
A_mh = torch.softmax(scores_mh, dim=-1)
context_mh = A_mh @ V_h                                      # (B, H, N, d_k)
print(f"  Attention: scores {scores_mh.shape}, weights {A_mh.shape}, context {context_mh.shape}")

# Concatenate heads
context_concat = context_mh.transpose(1, 2).contiguous().view(B, N, d_model)
print(f"  Concatenated: {context_concat.shape}")

# Output projection
output_mh = context_concat @ W_O
print(f"  Final output: {output_mh.shape}")

# ============================================================
# PART 6: Gradient Flow -- Outer Product Form
# ============================================================
print("\n" + "=" * 60)
print("PART 6: Gradient Flow (Outer Product)")
print("=" * 60)

# Forward with requires_grad
X_g = torch.randn(B, D, requires_grad=True)
W_g = torch.randn(D, H, requires_grad=True)
Y_g = X_g @ W_g
loss = Y_g.sum()
loss.backward()

# Verify gradient shape: dL/dW must match W shape
print(f"  W shape: {W_g.shape}")
print(f"  dL/dW shape: {W_g.grad.shape}")
print(f"  Match: {W_g.grad.shape == W_g.shape}")

# Manual outer product: dL/dW = X^T @ dL/dY
dL_dY = torch.ones(B, H)
dL_dW_manual = X_g.T @ dL_dY
print(f"  Manual dL/dW shape: {dL_dW_manual.shape}")
print(f"  Gradient matches manual: {torch.allclose(W_g.grad, dL_dW_manual)}")

# Gradient w.r.t. input has shape matching input
print(f"  X shape: {X_g.shape}")
print(f"  dL/dX shape: {X_g.grad.shape}")

# ============================================================
# PART 7: Common Pitfalls Demo
# ============================================================
print("\n" + "=" * 60)
print("PART 7: Common Pitfalls")
print("=" * 60)

# Pitfall: dim mismatch
print("\n  Pitfall 1 -- Dim mismatch:")
try:
    _ = torch.randn(4, 5) @ torch.randn(4, 6)
except RuntimeError as e:
    print(f"    {e}")

# Pitfall: non-contiguous
print("\n  Pitfall 2 -- Non-contiguous:")
x_nc = torch.randn(3, 4).T
print(f"    Transposed contiguous: {x_nc.is_contiguous()}")
try:
    _ = x_nc.view(-1)
except RuntimeError as e:
    print(f"    .view() fails: {e}")
x_fixed = x_nc.contiguous()
print(f"    After .contiguous(): {x_fixed.is_contiguous()}")
print(f"    .view() works: {x_fixed.view(-1).shape}")

# Pitfall: softmax wrong dim
print("\n  Pitfall 3 -- Softmax dim:")
logits_3d = torch.randn(2, 3, 10)
probs_batch = torch.softmax(logits_3d, dim=-1)
print(f"    Correct dim=-1: {probs_batch.shape}, rows sum={probs_batch[0,0].sum().item():.2f}")

print("\nAll demos complete.")
