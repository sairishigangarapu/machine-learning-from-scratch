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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\Omega$ | **Sample space** — set of all possible outcomes of a random experiment | Defines the universe of all possible results; every event is a subset of $\Omega$, and probabilities are assigned to subsets of $\Omega$ |
| $\{1, 2, 3, 4, 5, 6\}$ | The six possible outcomes of rolling a fair die | A concrete example of a finite sample space; each outcome is equally likely for a fair die |

**Event:** A subset of the sample space.

$$
E = \{2, 4, 6\} \quad \text{(rolling an even number)}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $E$ | **Event** — a subset of the sample space $\Omega$ | An event is a collection of outcomes we care about; in ML, events correspond to target outcomes (e.g., "email is spam") |
| $\{2, 4, 6\}$ | The event of rolling an even number on a die | Example of an event containing three outcomes; for a fair die, $P(E) = \frac{3}{6} = 0.5$ |

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

| Term | Definition | Significance |
|:---|:---|:---|
| $P(n, r)$ | **Permutation** — number of ways to arrange $r$ items from $n$ distinct items | Order matters; used when sequence is important (e.g., ranking, ordering features by importance) |
| $n!$ | **Factorial** — product of all integers from $1$ to $n$ | Counts the total number of ways to arrange $n$ distinct items |
| $n - r$ | Remaining items after selecting $r$ | The denominator removes arrangements of the unselected items |

### Combinations (Order doesn't matter)
$$
\binom{n}{r} = \frac{n!}{r!(n-r)!}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\binom{n}{r}$ | **Combination** — number of ways to choose $r$ items from $n$ regardless of order | Order does not matter; used for feature subset selection, measuring model complexity, and counting outcomes in probability |
| $n!$ | Factorial of $n$, the total number of items | Scales the count; larger $n$ creates more possible combinations |
| $r!(n-r)!$ | Denominator accounts for indistinguishable arrangements | Dividing by $r!$ removes the effect of ordering within the chosen set, and $(n-r)!$ removes ordering of unchosen items |

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

| Term | Definition | Significance |
|:---|:---|:---|
| $P(A \mid B)$ | **Conditional probability** — probability of $A$ given $B$ occurred | The foundation of Bayesian inference; enables updating beliefs when new evidence arrives |
| $P(A \cap B)$ | **Joint probability** — probability both $A$ and $B$ occur | Measures overlap between events; required to normalize the conditional probability |
| $P(B)$ | **Marginal probability** of $B$ | Normalizes the joint probability so the total conditional probability sums to 1; $P(B) > 0$ is required for the fraction to be defined |

**Intuition:** Given that $B$ has occurred, what is the probability that $A$ also occurs? We restrict our universe to $B$ and measure what fraction of $B$ overlaps with $A$.

**ML Connection:** Naive Bayes classifier directly uses conditional probability:

$$
P(\text{spam} | \text{words}) \propto P(\text{words} | \text{spam}) \cdot P(\text{spam})
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $P(\text{spam} \mid \text{words})$ | **Posterior** — probability of spam given observed words | The output of a classifier; what we want to compute for each incoming email |
| $P(\text{words} \mid \text{spam})$ | **Likelihood** — probability of seeing these words in a spam email | Captures the word distribution within the spam class; estimated from training data |
| $P(\text{spam})$ | **Prior** — overall proportion of spam emails | Captures baseline spam rate before seeing any content; can be set from historical data |
| $\propto$ | **Proportional to** — the denominator is omitted | The denominator $P(\text{words})$ is constant across classes, so it cancels when comparing which class is most likely |

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

| Term | Definition | Significance |
|:---|:---|:---|
| $P(A \cap B)$ | **Joint probability** of $A$ and $B$ | Under independence, the joint factorizes into the product of marginals — a massive computational simplification |
| $P(A)$ | Marginal probability of $A$ | Probability of $A$ regardless of $B$ |
| $P(B)$ | Marginal probability of $B$ | Probability of $B$ regardless of $A$ |
| $P(A) \cdot P(B)$ | Product of individual probabilities | For independent events, $P(A \cap B) = P(A)P(B)$; the **Naive Bayes** classifier exploits this even $= \text{when features are not truly independent}$ |

Equivalently: $P(A | B) = P(A)$ — knowing $B$ occurred doesn't change the probability of $A$.

**ML Connection:** The **Naive Bayes** classifier assumes all features are conditionally independent given the class label. This is almost always wrong in reality, but the classifier still works remarkably well in practice (spam filtering, text classification).

---

## 6. Total Probability Law

If $B_1, B_2, \dots, B_n$ form a **partition** of $\Omega$ (disjoint, exhaustive):

$$
P(A) = \sum_{i=1}^{n} P(A | B_i) P(B_i)
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $P(A)$ | **Total probability** of event $A$ | Decomposes a complex probability into weighted sums over a partition of the sample space |
| $P(A \mid B_i)$ | Conditional probability of $A$ given partition element $B_i$ | Each $B_i$ represents a distinct scenario; $P(A \mid B_i)$ is how likely $A$ is in that scenario |
| $P(B_i)$ | **Prior probability** of $B_i$ | Weights each scenario by how likely it is to occur; the sum of all $P(B_i)$ must equal $1$ |
| $\sum_{i=1}^{n}$ | Sum over all $n$ partition elements | The law of total probability marginalizes out the conditioning variable — critical for computing $P(A)$ when $A$ depends on another variable |

**ML Connection:** This is how mixture models work. A Gaussian Mixture Model represents the data distribution as a weighted sum of component Gaussians:

$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \Sigma_k)
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $p(\mathbf{x})$ | **Probability density** of data point $\mathbf{x}$ | The overall distribution is a weighted mixture; each data point could have come from any component |
| $\pi_k$ | **Mixing weight** for component $k$ | Sums to 1 over all $k$; represents the prior probability that a data point belongs to component $k$ |
| $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \Sigma_k)$ | **Gaussian density** for component $k$ with mean $\boldsymbol{\mu}_k$ and covariance $\Sigma_k$ | Each mixture component is a Gaussian; the overall density is a convex combination of them |
| $K$ | Number of mixture components | Model hyperparameter; too few underfits, too many overfits |

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

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 47: Sets and Basic Operations](Lecture%2047%20Sets%20and%20Basic%20Operations.md) — Foundational set theory for probability spaces
- **Next:** [Lecture 49: Bayes Theorem and Random Variables](Lecture%2049%20Bayes%20Theorem%20and%20Random%20Variables.md) — Extends probability with conditional reasoning
- **Related:** [Lecture 49: Bayes Theorem and Random Variables](Lecture%2049%20Bayes%20Theorem%20and%20Random%20Variables.md) — Direct application of probability concepts
- **Related:** [Lecture 24: Logistic Regression-I](Lecture%2024%20Logistic%20Regression-I.md) — Uses probability for classification
