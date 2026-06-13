## L25: Reinforcement Learning from Human Feedback (RLHF)

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. RLHF Pipeline Overview

### Motivation and Intuition

Large language models (LLMs) trained on next-token prediction learn statistical patterns in text, but they do not inherently produce outputs that are helpful, harmless, or honest. RLHF steers the model toward human-preferred responses by first learning a reward model from human comparisons, then using reinforcement learning to maximize that reward while staying close to the original model.

The pipeline has three stages:

1. **Supervised Fine-Tuning (SFT)** — Fine-tune the pretrained LLM on high-quality human demonstrations (prompt → ideal response) so it learns the desired style and format.
2. **Reward Model Training** — Collect human preferences (A vs B comparisons) and train a reward model to predict which response humans prefer.
3. **Proximal Policy Optimization (PPO)** — Use the reward model as a proxy for human preference and fine-tune the SFT model with PPO, adding a KL penalty to prevent the policy from drifting too far from the SFT model.

---

## 2. Supervised Fine-Tuning (SFT)

### Motivation and Intuition

The pretrained base model predicts the next token from internet text. SFT teaches it to respond in a conversational format: given a prompt, produce a helpful response. This is standard supervised learning — minimize cross-entropy loss on human-written prompt–response pairs.

$$
\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{SFT}}} \left[ \sum_{t=1}^{|y|} \log \pi_{\text{SFT}}(y_t \mid x, y_{<t}) \right]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $x$ | Input prompt | Conditions the model on the user's request |
| $y$ | Human-written response | Provides the target distribution to imitate |
| $\pi_{\text{SFT}}$ | Policy (model) after SFT | Starting point for reward modeling and RLHF |

---

## 3. Reward Model Training

### Motivation and Intuition

Human preferences are expensive and slow to collect at inference time. Instead, we train a reward model $R_\phi$ that takes a response and outputs a scalar score. Humans are shown two responses to the same prompt and pick the better one. The reward model learns to assign higher scores to preferred responses.

The Bradley-Terry preference model gives a probability that response $y_a$ is preferred over $y_b$:

$$
P(y_a \succ y_b \mid x) = \frac{\exp(R_\phi(x, y_a))}{\exp(R_\phi(x, y_b)) + \exp(R_\phi(x, y_a))}
$$

We minimize the negative log-likelihood of the observed preferences:

$$
\mathcal{L}_R = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[ \log \sigma \big( R_\phi(x, y_w) - R_\phi(x, y_l) \big) \right]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $y_w$ | The preferred (winning) response | Human judged this response better |
| $y_l$ | The dispreferred (losing) response | Human judged this response worse |
| $R_\phi(x, y)$ | Scalar reward for response $y$ given prompt $x$ | Proxy for human preference |
| $\sigma$ | Sigmoid function | Squashes reward difference into (0, 1) |
| $\mathcal{D}_{\text{pref}}$ | Dataset of human comparisons | Usually thousands to millions of pairwise labels |

The reward model is typically initialized from the SFT model with the final unembedding layer replaced by a single linear layer that outputs a scalar.

---

## 4. Proximal Policy Optimization (PPO) with KL Penalty

### Motivation and Intuition

Now we fine-tune the SFT model $\pi_{\text{SFT}}$ to maximize the reward model's score. But if we optimize too aggressively, the model may find adversarial responses that score high but are nonsensical or degenerate. To prevent this, we add a KL divergence penalty that keeps the learned policy $\pi_\theta$ close to $\pi_{\text{SFT}}$.

The full PPO objective for RLHF:

$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}_{(x, y) \sim \pi_\theta} \left[ R_\phi(x, y) \right] - \beta \cdot \mathbb{E}_{x \sim \mathcal{D}} \left[ D_{\text{KL}} \big( \pi_\theta(\cdot \mid x) \;\|\; \pi_{\text{SFT}}(\cdot \mid x) \big) \right]
$$

In practice, we use the clipped PPO surrogate objective from Schulman et al. 2017. For each token $t$:

$$
\mathcal{L}^{\text{PPO-clip}} = \mathbb{E} \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{SFT}}(y_t \mid x, y_{<t})}$ is the probability ratio and $\hat{A}_t$ is the advantage estimate (typically $R_\phi(x, y) - \text{baseline}$).

The combined RLHF PPO loss with KL penalty:

$$
\mathcal{L}_{\text{RLHF}} = \mathcal{L}^{\text{PPO-clip}} - \beta \cdot D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{SFT}})
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\pi_\theta$ | Policy being optimized | The model we are aligning |
| $\pi_{\text{SFT}}$ | Frozen SFT model | Reference distribution to stay near |
| $\beta$ | KL penalty coefficient | Controls how tightly we constrain the policy |
| $D_{\text{KL}}(P \parallel Q)$ | Kullback–Leibler divergence | Measures how much $P$ diverges from $Q$ |
| $r_t(\theta)$ | Importance sampling ratio | Corrects for off-policy sampling |
| $\epsilon$ | Clipping hyperparameter | Stabilizes PPO (typically 0.2) |
| $\hat{A}_t$ | Estimated advantage | How much better this action is than average |

---

### Why RLHF Aligns LLMs

| Factor | Explanation |
| :--- | :--- |
| **Human preferences** | Reward model encodes nuanced human values that are hard to specify in a loss function |
| **KL constraint** | Prevents the model from exploiting the reward model or forgetting its language abilities |
| **PPO stability** | Clipping and advantage normalization make policy gradient training stable for large models |
| **Iterative process** | The reward model can be updated with new preferences and the cycle repeated, enabling continuous improvement |

---

> **Check your intuition:** Why is the KL penalty against the SFT model necessary instead of just maximizing the reward? What would happen if $\beta = 0$?

---

## Prerequisites and Further Reading

- **StatQuest:** Reinforcement Learning with Neural Networks (L23), Reinforcement Learning Mathematical Details (L24), Decoder-Only Transformers (L21)
- **Papers:** Schulman et al., "Proximal Policy Optimization Algorithms" (2017); Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022)
- **Concepts:** Policy gradients, cross-entropy loss, KL divergence, transformer architectures
