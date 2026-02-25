# Lecture 09: Eigenvalues and Eigenvectors

## Introduction
This lecture covers the fundamental concepts of eigenvalues and eigenvectors in linear algebra, which are crucial in various fields including machine learning.

## Eigenvalues and Eigenvectors
Eigenvalues and eigenvectors of a matrix A are defined as:

If \( A \) is a square matrix, then an eigenvector \( \mathbf{v} \) is a non-zero vector such that:
\[ A \mathbf{v} = \lambda \mathbf{v} \]
where \( \lambda \) is a scalar known as the eigenvalue.

### Matrix Representation
For a matrix \( A \), the eigenvalue equation can be represented in terms of the matrix \( A - \lambda I \):
\[ \begin{pmatrix}
  a & b \\
  c & d
\end{pmatrix} \begin{pmatrix}
  v_1 \\
  v_2
\end{pmatrix} = \lambda \begin{pmatrix}
  v_1 \\
  v_2
\end{pmatrix} \]

### Conclusion
Understanding eigenvalues and eigenvectors is essential for grasping many more advanced topics in machine learning, including PCA (Principal Component Analysis).