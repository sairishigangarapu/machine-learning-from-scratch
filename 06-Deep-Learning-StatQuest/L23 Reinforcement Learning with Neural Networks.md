## Reinforcement Learning with Neural Networks

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. What Is Reinforcement Learning?

### Motivation and Intuition

In supervised learning, we have input-output pairs and train via backpropagation by comparing predictions to ground truth. In reinforcement learning (RL), we don't know the correct output in advance. The model (agent) must interact with an environment, take actions, and receive feedback in the form of rewards. The goal is to learn a **policy** — a strategy for selecting actions — that maximizes cumulative reward.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Agent | The decision-maker (neural network) | Takes actions based on current state |
| Environment | The world the agent interacts with | Returns next state and reward after each action |
| State ($s$) | Current configuration of the environment | Input to the policy network |
| Action ($a$) | A choice the agent makes | Output of the policy network |
| Reward ($r$) | Scalar signal indicating how good the last action was | Positive = reinforce, negative = discourage |

### Key Idea

The agent guesses an action, observes the outcome, and updates its policy based on the reward received. This is fundamentally different from supervised learning where the correct answer is known upfront.

---

## 2. Q-Learning

### Motivation and Intuition

Q-learning learns a **Q-function** $Q(s, a)$ that estimates the expected cumulative reward of taking action $a$ in state $s$. The optimal policy is to pick the action with the highest Q-value.

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $Q(s, a)$ | Current estimate of value of action $a$ in state $s$ | What the agent thinks the action is worth |
| $\alpha$ | Learning rate | How much to update the estimate (0 to 1) |
| $r$ | Reward received after taking action $a$ | Immediate feedback from environment |
| $\gamma$ | Discount factor (0 to 1) | Balances immediate vs. future rewards |
| $\max_{a'} Q(s', a')$ | Best Q-value in next state $s'$ | Bootstraps from future estimate |

### DQN (Deep Q-Network)

DQN uses a neural network to approximate $Q(s, a)$ for large or continuous state spaces (such as images). The network takes state as input and outputs Q-values for each action.

---

## 3. Policy Gradients

### Motivation and Intuition

Instead of learning Q-values and deriving a policy, policy gradients optimize the policy $\pi_\theta(a|s)$ directly. The neural network takes state as input and outputs action probabilities.

The key insight: we don't have target labels, but we can still compute gradients. We guess an action, compute the gradient direction that would increase its probability, then multiply by the actual reward. Positive reward = reinforce the guess. Negative reward = flip the gradient direction, making the action less likely.

$$
\nabla_\theta J(\theta) \approx \mathbb{E}_{\pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a|s) \cdot R \right]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\pi_\theta(a|s)$ | Policy network parameterized by $\theta$ | Probability of action $a$ in state $s$ |
| $\nabla_\theta \log \pi_\theta(a|s)$ | Score function gradient | Direction to increase probability of action $a$ |
| $R$ | Cumulative reward after taking action $a$ | Scales gradient: positive = reinforce, negative = reverse |

```python
import numpy as np

class PolicyNetwork:
    def __init__(self, input_dim, output_dim, hidden_dim=16):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)

    def forward(self, state):
        hidden = np.tanh(state @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def sample_action(self, state):
        probs = self.forward(state)
        return np.random.choice(len(probs), p=probs)
```

---

## 4. The REINFORCE Algorithm

### Motivation and Intuition

REINFORCE (Monte Carlo policy gradient) collects full episodes and uses the cumulative return from each time step to update the policy.

Steps:
1. Run the policy to collect an episode: $(s_0, a_0, r_0), (s_1, a_1, r_1), \dots$
2. For each time step $t$, compute the discounted return $G_t$ from that point onward.
3. Compute $\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$
4. Update policy parameters via gradient ascent.

$$
\nabla_\theta J(\theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t
$$

$$
G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $G_t$ | Discounted return from time step $t$ | How good the future turned out after action $a_t$ |
| $\gamma$ | Discount factor | Controls how much we care about future vs. immediate rewards |
| $T$ | Episode length | Number of steps until terminal state |

```python
def reinforce(policy_net, episodes, gamma=0.99, learning_rate=0.01):
    for episode in episodes:
        states, actions, rewards = episode

        # Compute discounted returns
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        # Update policy for each step
        for state, action, G_t in zip(states, actions, returns):
            probs = policy_net.forward(state)
            d_log_p = np.zeros_like(probs)
            d_log_p[action] = -1.0 / probs[action]  # gradient of -log prob
            gradient = d_log_p * G_t
            policy_net.W2 -= learning_rate * gradient  # simplified update
```

---

## 5. Exploration vs. Exploitation

### Motivation and Intuition

A fundamental RL challenge: should the agent try new actions (exploration) or use known good actions (exploitation)?

The policy network's softmax output naturally provides exploration — even when one action is strongly preferred, others have nonzero probability. **Temperature** controls this balance:

$$
\pi_\theta(a|s) = \frac{e^{z_a / \tau}}{\sum_{a'} e^{z_{a'} / \tau}}
$$

| $\tau$ | Effect |
| :--- | :--- |
| High ($>1$) | More uniform probabilities — more exploration |
| Low ($<1$) | Sharper distribution — more exploitation |
| $\tau \to 0$ | Greedy — always pick most likely action |

```python
def softmax_with_temperature(logits, temperature=1.0):
    logits = logits / temperature
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)
```

---

> **Check your intuition:** In policy gradients, the gradient tells us how to increase the probability of the action we just took. If the reward is negative, we multiply the gradient by a negative number. Why does this flip the gradient in the correct direction — making the bad action less likely next time?

---

## Prerequisites and Further Reading

- [StatQuest: Neural Networks and Backpropagation](https://www.youtube.com/watch?v=w8yWXqWQYmU)
- [StatQuest: Gradient Descent](https://www.youtube.com/watch?v=sDv4f4s2SB8)
- Sutton & Barto, "Reinforcement Learning: An Introduction"
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013) — DQN paper
