## Reinforcement Learning Mathematical Details

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. The Goal: Maximize Expected Return

### Motivation and Intuition

The fundamental goal of reinforcement learning is to find a policy $\pi_\theta(a|s)$ that maximizes the expected cumulative discounted reward (the **expected return**). The policy is a neural network parameterized by $\theta$ (weights and biases). We need to compute how to adjust $\theta$ to increase the expected return — this requires the gradient of the expected return with respect to $\theta$.

$$
J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $J(\theta)$ | Expected return under policy $\pi_\theta$ | The objective we maximize |
| $\mathbb{E}_{\pi_\theta}$ | Expectation over trajectories from following $\pi_\theta$ | Average over all possible sequences of states, actions, rewards |
| $\gamma$ | Discount factor ($0 \leq \gamma < 1$) | Ensures convergence for infinite horizons; trades off immediate vs. future reward |
| $r_t$ | Reward at time step $t$ | Scalar feedback from the environment |

---

## 2. The Policy Gradient Theorem

### Motivation and Intuition

We cannot directly differentiate $J(\theta)$ because it involves an expectation over unknown environment dynamics. The policy gradient theorem gives us a tractable expression for $\nabla_\theta J(\theta)$ that depends only on the policy, not on the environment model.

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s, a) \right]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\nabla_\theta J(\theta)$ | Gradient of expected return w.r.t. policy parameters | Tells us how to adjust $\theta$ to increase expected return |
| $\nabla_\theta \log \pi_\theta(a|s)$ | Score function — gradient of log-probability of action $a$ | Direction in parameter space that increases $a$'s probability |
| $Q^{\pi_\theta}(s, a)$ | Action-value function: expected return from taking $a$ in $s$ then following $\pi_\theta$ | How good action $a$ actually is — scales the gradient |

> **Key insight:** The gradient is an expectation over the product of two terms: the score function (which doesn't depend on reward) and the Q-value (which captures reward information). The environment dynamics cancel out because they don't depend on $\theta$ — this is the "log-derivative trick."

### Why the Environment Disappears

The probability of a trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$ is:

$$
P(\tau) = P(s_0) \prod_{t} \pi_\theta(a_t|s_t) \cdot P(s_{t+1}, r_t | s_t, a_t)
$$

Taking the log and differentiating w.r.t. $\theta$:

$$
\nabla_\theta \log P(\tau) = \sum_{t} \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

The environment dynamics $P(s_{t+1}, r_t | s_t, a_t)$ do not depend on $\theta$, so they differentiate to zero. This is why policy gradients work without knowing the environment model.

---

## 3. REINFORCE Derivation

### Motivation and Intuition

REINFORCE replaces the true $Q^{\pi_\theta}(s, a)$ with the Monte Carlo return $G_t$ (the actual cumulative reward observed from that point forward in an episode). This gives us a practical algorithm.

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]
$$

### Step-by-Step Derivation

**Step 1:** Start with the policy gradient theorem:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s, a) \right]
$$

**Step 2:** Replace $Q^{\pi_\theta}(s, a)$ with the return $G_t$ (an unbiased sample):

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T_i} \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) \cdot G_{i,t}
$$

**Step 3:** For a single episode, the update for each time step $t$ is:

$$
\theta \leftarrow \theta + \alpha \cdot G_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\alpha$ | Learning rate | Step size for gradient ascent |
| $G_t$ | Discounted return from step $t$ | $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ |
| $\nabla_\theta \log \pi_\theta(a_t|s_t)$ | Direction to increase $a_t$'s probability | Scaled by $G_t$ — positive return = reinforce, negative = discourage |

```python
import numpy as np

def reinforce_update(policy, episode, gamma=0.99, alpha=0.01):
    states, actions, rewards = episode

    # 1. Compute discounted returns G_t
    T = len(rewards)
    returns = np.zeros(T)
    G = 0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G

    # 2. Update policy for each step
    for state, action, G_t in zip(states, actions, returns):
        probs = policy.forward(state)     # π_θ(a|s)
        # Cross-entropy loss gradient for chosen action
        d_log_p = probs.copy()
        d_log_p[action] -= 1.0            # ∇_θ log π_θ(a|s) for softmax

        # Gradient ascent: θ += α * G_t * ∇_θ log π_θ(a|s)
        # (simplified — real code would backprop through the network)
        policy.W2 -= alpha * G_t * d_log_p  # direction depends on sign of G_t
```

---

## 4. The Advantage Function and Baselines

### Motivation and Intuition

The returns $G_t$ in REINFORCE have high variance — the same state-action pair can lead to very different returns depending on later randomness. A **baseline** $b(s)$ subtracts a state-dependent value from $G_t$ without changing the expected gradient, reducing variance.

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot \left( G_t - b(s) \right) \right]
$$

The optimal baseline is the state-value function $V^{\pi_\theta}(s)$, giving the **advantage function**:

$$
A^{\pi_\theta}(s, a) = Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $b(s)$ | Baseline — can be any function of state (or constant) | Reduces variance without introducing bias |
| $V^{\pi_\theta}(s)$ | State-value function: expected return from state $s$ | "How good is this state on average?" |
| $A^{\pi_\theta}(s, a)$ | Advantage: how much better action $a$ is than average | Positive = better than average action, negative = worse |

> **Intuition:** An action might get a return of +50, but if the baseline is +50 (the state is already great), the advantage is 0 — the action wasn't special. Another action gets +10, but the baseline is -30, so the advantage is +40 — that action was much better than average and should be reinforced.

---

## 5. Actor-Critic Methods

### Motivation and Intuition

Actor-critic methods use two neural networks:

- **Actor** (policy network $\pi_\theta$): decides which actions to take.
- **Critic** (value network $V_\phi$): estimates the value function to provide a baseline and reduce variance.

The critic learns $V_\phi(s)$ by minimizing the mean squared error between its prediction and the actual return:

$$
\mathcal{L}_{critic} = \mathbb{E} \left[ \left( G_t - V_\phi(s_t) \right)^2 \right]
$$

The actor uses the advantage estimated by the critic:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot \left( G_t - V_\phi(s) \right) \right]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Actor $\pi_\theta$ | Policy network — outputs action probabilities | What we're optimizing |
| Critic $V_\phi$ | Value network — estimates state value | Provides baseline to reduce policy gradient variance |
| TD error | $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ | Alternative advantage estimate using bootstrapping |

```python
class ActorCritic:
    def __init__(self, state_dim, action_dim, hidden_dim=16):
        # Actor: policy network
        self.actor_W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.actor_W2 = np.random.randn(hidden_dim, action_dim) * 0.1
        # Critic: value network
        self.critic_W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.critic_W2 = np.random.randn(hidden_dim, 1) * 0.1

    def forward(self, state):
        # Actor outputs action probabilities
        hidden = np.tanh(state @ self.actor_W1)
        logits = hidden @ self.actor_W2
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)

        # Critic outputs state value V(s)
        c_hidden = np.tanh(state @ self.critic_W1)
        value = c_hidden @ self.critic_W2  # scalar
        return probs, value.item()

    def update(self, state, action, reward, next_state, gamma=0.99, alpha=0.01):
        probs, value = self.forward(state)
        _, next_value = self.forward(next_state)

        # TD error: advantage estimate
        td_error = reward + gamma * next_value - value

        # Actor update (policy gradient with advantage)
        d_log_p = probs.copy()
        d_log_p[action] -= 1.0
        self.actor_W2 -= alpha * td_error * d_log_p  # simplified

        # Critic update (value function regression)
        self.critic_W2 -= alpha * (-td_error)  # simplified
```

---

## 6. Gradient of the Policy (Softmax Case)

### Motivation and Intuition

For a discrete action space with softmax policy, the gradient of the log-probability is:

$$
\nabla_\theta \log \pi_\theta(a|s) = \nabla_\theta z_a - \sum_{a'} \pi_\theta(a'|s) \nabla_\theta z_{a'}
$$

where $z_a$ is the logit (raw score) for action $a$ before softmax.

This simplifies to: the gradient is the input features (hidden state) multiplied by $(1 - \pi_\theta(a|s))$ for the chosen action, and by $-\pi_\theta(a'|s)$ for all other actions.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $z_a$ | Logit for action $a$ | Raw output of the network before softmax |
| $\nabla_\theta z_a$ | Gradient of logit $a$ w.r.t. parameters | How small changes in $\theta$ affect action $a$'s preference |

---

> **Check your intuition:** In REINFORCE, suppose an action leads to a very positive return $G_t = 100$, but the baseline $b(s) = 95$. The advantage is only 5. Another action gets $G_t = 20$ with baseline $b(s) = -10$, so the advantage is 30. Which action gets reinforced more strongly? Why does the baseline make this correct?

---

## Prerequisites and Further Reading

- [StatQuest: Reinforcement Learning with Neural Networks (Essential Concepts)](https://www.youtube.com/watch?v=9hbQieQh7-o)
- [StatQuest: Gradient Descent](https://www.youtube.com/watch?v=sDv4f4s2SB8)
- [StatQuest: Backpropagation](https://www.youtube.com/watch?v=i94OvYb6noo)
- Sutton & Barto, "Reinforcement Learning: An Introduction" — Chapters 13 (Policy Gradient Methods)
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017) — modern policy gradient method
- Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (A3C, 2016)
- Lilian Weng's Policy Gradient blog: https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
