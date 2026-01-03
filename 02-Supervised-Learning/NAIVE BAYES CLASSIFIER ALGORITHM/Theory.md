# Naive Bayes Classifier 🧠

## 1. Concept Overview
**Naive Bayes** is a probabilistic classifier based on **Bayes' Theorem**. It is called "Naive" because it makes a strong assumption: **conditional independence** between features (i.e., the presence of one feature does not affect the others).

Despite this "naive" assumption, it works remarkably well for:
* **Spam Filtering** (Text Classification)
* **Sentiment Analysis**
* **Recommendation Systems**

---

## 2. The Mathematics: Bayes' Theorem

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$



| Term | Definition | Context (Spam Example) |
| :--- | :--- | :--- |
| **$P(A \mid B)$** | **Posterior** | Prob. email is Spam given word "Lottery" exists. |
| **$P(B \mid A)$** | **Likelihood** | Prob. word "Lottery" appears in Spam emails. |
| **$P(A)$** | **Prior** | Prob. any random email is Spam. |
| **$P(B)$** | **Evidence** | Prob. word "Lottery" appears in *any* email. |

---

## 3. Types of Naive Bayes Classifiers

| Type | Data Assumption | Use Case |
| :--- | :--- | :--- |
| **Bernoulli NB** | Binary (0/1) | Word presence/absence (True/False). |
| **Multinomial NB** | Discrete Counts | Word frequency (Bag of Words). |
| **Gaussian NB** | Continuous (Normal Dist.) | Numerical features (e.g., Iris Dataset). |

---

## 4. Feature Extraction: Bag of Words (BoW)
Machine Learning models cannot process raw text. We use **Count Vectorization** to turn text into numbers.

**Example Corpus:**
1. "I love NLP"
2. "NLP is fun"

**Document-Term Matrix:**
| I | love | NLP | is | fun |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1 | 1 | 0 | 0 |
| 0 | 0 | 1 | 1 | 1 |

---
**External Exercise:** [Codebasics Naive Bayes Lab](https://github.com/codebasics/py/blob/master/ML/14_naive_bayes/14_naive_bayes_titanic.ipynb)
