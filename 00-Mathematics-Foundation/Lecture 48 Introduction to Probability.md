## Introduction to Probability

*Essential Mathematics for ML — Structured Notes*

---

## 1. Why Probability?

### Motivation and Intuition
Machine learning is fundamentally about making predictions under uncertainty. Will this email be spam? Is this tumor malignant? What will the stock price be tomorrow? We don't have deterministic answers — we have **beliefs** and **evidence**. Probability theory is the mathematical framework for reasoning about uncertainty, and it is the foundation of every probabilistic ML model: Naive Bayes, Hidden Markov Models, Gaussian Mixture Models, VAEs, and more.

### Definitions

**Random Experiment:** A process with an uncertain outcome (rolling a die, flipping a coin, measuring a feature).

**Sample Space ($\Omega$):** The set of all possible outcomes.

$$
\Omega = \{1, 2, 3, 4, 5, 6\} \quad \text{(rolling a fair die)}
$$

**Event:** A subset of the sample space.

$$
E = \{2, 4, 6\} \quad \text{(rolling an even number)}
$$

**Probability ($P$):** A function $P: \mathcal{F} \to [0, 1]$ that assigns a number between 0 and 1 to each event, satisfying:

1. $P(\Omega) = 1$ (something must happen)
2. $P(E) \ge 0$ for all events $E$
3. For disjoint events $E_1, E_2, \dots$: $P(\bigcup E_i) = \sum P(E_i)$ (additivity)

---

## 2. Combinatorics: Counting Outcomes

### Permutations (Order matters)
$$
P(n, r) = \frac{n!}{(n-r)!}
$$

### Combinations (Order doesn't matter)
$$
\binom{n}{r} = \frac{n!}{r!(n-r)!}
$$

**ML Connection:** Feature selection — choosing $k$ features from $n$ candidates involves $\binom{n}{k}$ possible subsets. For $n = 100$ features and $k = 10$, that's $\sim 1.7 \times 10^{13}$ subsets. Exhaustive search is impossible.

```python
import math

# Number of ways to choose 3 features from 10
print(math.comb(10, 3))  # 120

# Number of ways to arrange 5 items
print(math.perm(5, 5))   # 120
```

---

## 3. Axioms of Probability

| Axiom | Statement |
|:---|:---|
| **Non-negativity** | $P(E) \ge 0$ |
| **Normalization** | $P(\Omega) = 1$ |
| **Additivity** | $P(E_1 \cup E_2) = P(E_1) + P(E_2)$ if $E_1 \cap E_2 = \emptyset$ |

### Derived Properties

* $P(\emptyset) = 0$
* $P(E^c) = 1 - P(E)$
* $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ (Inclusion-Exclusion)
* $P(A) \le 1$

---

## 4. Conditional Probability

$$
P(A | B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0
$$

**Intuition:** Given that $B$ has occurred, what is the probability that $A$ also occurs? We restrict our universe to $B$ and measure what fraction of $B$ overlaps with $A$.

**ML Connection:** Naive Bayes classifier directly uses conditional probability:

$$
P(\text{spam} | \text{words}) \propto P(\text{words} | \text{spam}) \cdot P(\text{spam})
$$

```python
# P(spam | contains "free") = P(contains "free" AND spam) / P(spam)
# If 100 emails, 30 spam, 20 of spam contain "free"
p_spam = 30 / 100
p_free_and_spam = 20 / 100
p_free = 40 / 100  # 40 emails contain "free"

p_spam_given_free = p_free_and_spam / p_free
print(f"P(spam | 'free') = {p_spam_given_free:.2f}")  # 0.50
```

---

## 5. Independence

Events $A$ and $B$ are **independent** if:

$$
P(A \cap B) = P(A) \cdot P(B)
$$

Equivalently: $P(A | B) = P(A)$ — knowing $B$ occurred doesn't change the probability of $A$.

**ML Connection:** The **Naive Bayes** classifier assumes all features are conditionally independent given the class label. This is almost always wrong in reality, but the classifier still works remarkably well in practice (spam filtering, text classification).

---

## 6. Total Probability Law

If $B_1, B_2, \dots, B_n$ form a **partition** of $\Omega$ (disjoint, exhaustive):

$$
P(A) = \sum_{i=1}^{n} P(A | B_i) P(B_i)
$$

**ML Connection:** This is how mixture models work. A Gaussian Mixture Model represents the data distribution as a weighted sum of component Gaussians:

$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \Sigma_k)
$$

where $\pi_k = P(\text{component } k)$ are the mixing weights.

---

## 7. Why Sets and Probability Matter Together

The connection is direct:
* A **set** defines what outcomes are possible.
* **Probability** assigns likelihoods to subsets of outcomes.
* **Events** are subsets, and probability measures on sets enable inference.

| Set Concept | Probability Concept |
|:---|:---|
| Sample space $\Omega$ | All possible outcomes |
| Event $E \subseteq \Omega$ | A subset with a probability |
| $A \cap B$ | Both events occur |
| $A \cup B$ | At least one occurs |
| $A^c$ | Event does not occur |
| $A \subseteq B$ | $A$ implies $B$ |

> **Check your intuition:** If you flip a fair coin 3 times, what is the probability of getting at least one head? *(Answer: $P(\text{at least 1 H}) = 1 - P(\text{all tails}) = 1 - (1/2)^3 = 7/8$. Using the complement is often easier than summing individual cases.)*
