# Reinforcement Learning: From Policy Gradients to RLHF

## 1. RL Basics — Agent, Environment, State, Action, Reward

### Motivation and Intuition

In supervised learning, we have input-output pairs and train via backpropagation by comparing predictions to ground truth. In reinforcement learning (RL), we do not know the correct output in advance. The model (agent) must interact with an environment, take actions, and receive feedback in the form of rewards. The goal is to learn a **policy** — a strategy for selecting actions — that maximizes cumulative reward.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Agent | The decision-maker (neural network) | Takes actions based on current state |
| Environment | The world the agent interacts with | Returns next state and reward after each action |
| State ($s$) | Current configuration of the environment | Input to the policy network |
| Action ($a$) | A choice the agent makes | Output of the policy network |
| Reward ($r$) | Scalar signal indicating how good the last action was | Positive = reinforce behavior, negative = discourage |
| Policy ($\pi$) | Strategy mapping states to actions | What we aim to learn |
| Episode | A complete trajectory from start to terminal state | One trial (e.g., one game from start to game over) |
| Return ($G_t$) | Cumulative discounted reward from step $t$ onward | The total reward the agent actually accrues |

### Key Idea

The agent guesses an action, observes the outcome, and updates its policy based on the reward received. This is fundamentally different from supervised learning where the correct answer is known upfront.

**ML Connection:** RL powers game-playing AI (AlphaGo, Dota 2, StarCraft II), robotics (grasping, navigation), autonomous driving, and LLM alignment via RLHF (used in ChatGPT, Claude). It is the only learning paradigm that handles sequential decision-making with delayed rewards.

---

## 2. Markov Decision Process (MDP) and Return

### Motivation and Intuition

RL problems are formalized as **Markov Decision Processes** (MDPs). An MDP is defined by the tuple $(S, A, P, R, \gamma)$ where:
- $S$: set of states
- $A$: set of actions
- $P(s' \mid s, a)$: transition probability — dynamics of the environment
- $R(s, a)$: reward function
- $\gamma$: discount factor

The **Markov property** states that the future depends only on the current state and action, not on the history.

### Discounted Return

The return at time $t$ is the sum of discounted future rewards:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

| Symbol | Definition | Significance |
| :--- | :--- | :--- |
| $G_t$ | Discounted return from time step $t$ | The cumulative reward the agent aims to maximize |
| $r_{t+k}$ | Reward received at time $t+k$ | Scalar feedback at each step |
| $\gamma$ | Discount factor ($0 \leq \gamma < 1$) | Balances immediate vs. future rewards; ensures convergence for infinite horizons |

**Why discount?** (1) Future rewards are less certain. (2) Mathematically ensures finite returns for infinite-horizon problems. (3) Captures human preference for immediate rewards.

### The Objective

The agent's goal is to find a policy $\pi_\theta(a|s)$ that maximizes the **expected return**:

$$
J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $J(\theta)$ | Expected return under policy $\pi_\theta$ | The objective we maximize |
| $\mathbb{E}_{\pi_\theta}$ | Expectation over trajectories from following $\pi_\theta$ | Average over all possible sequences of states, actions, rewards |
| $T$ | Episode horizon (can be $\infty$) | Time step at which episode ends |

---

## 3. Q-Learning and DQN

### Motivation and Intuition

Q-learning learns a **Q-function** $Q(s, a)$ that estimates the expected cumulative reward of taking action $a$ in state $s$ and following the optimal policy thereafter. The optimal policy is to pick the action with the highest Q-value.

### Q-Learning Update

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $Q(s, a)$ | Current estimate of value of action $a$ in state $s$ | What the agent thinks the action is worth |
| $\alpha$ | Learning rate | How much to update the estimate (0 to 1) |
| $r$ | Reward received after taking action $a$ | Immediate feedback from environment |
| $\gamma$ | Discount factor (0 to 1) | Balances immediate vs. future rewards |
| $\max_{a'} Q(s', a')$ | Best Q-value in next state $s'$ | Bootstraps from the future optimal value |
| $r + \gamma \max_{a'} Q(s', a')$ | TD target | The "improved estimate" we move toward |

### Deep Q-Network (DQN)

DQN uses a neural network $Q_\phi(s, a)$ to approximate the Q-function for large or continuous state spaces (such as images). The network takes state as input and outputs Q-values for each action.

Key innovations in DQN:
- **Experience replay**: Store transitions $(s, a, r, s')$ in a buffer and sample randomly to break correlation
- **Target network**: Use a separate network for TD target computation, updated slowly for stability

$$
\mathcal{L}_{DQN} = \mathbb{E}_{(s, a, r, s') \sim \text{replay}} \left[ \left( r + \gamma \max_{a'} Q_{\phi^-}(s', a') - Q_\phi(s, a) \right)^2 \right]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $Q_{\phi^-}$ | Target network (frozen parameters) | Fixed target prevents chasing a moving target |
| Experience replay buffer | Memory of past transitions | Breaks temporal correlation, improves sample efficiency |

**ML Connection:** DQN achieved human-level performance on 49 Atari games using only raw pixels as input (Mnih et al., 2013). It was the first major success of deep RL.

---

## 4. Policy Gradients — REINFORCE Algorithm

### Motivation and Intuition

Instead of learning Q-values and deriving a policy, policy gradients optimize the policy $\pi_\theta(a|s)$ directly. The neural network takes state as input and outputs action probabilities.

The key insight: we do not have target labels, but we can still compute gradients through the **log-derivative trick**. We run an episode, observe which actions led to good returns, and increase their probability. Actions that led to poor returns are made less likely.

### The Policy Gradient Theorem

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s, a) \right]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\nabla_\theta J(\theta)$ | Gradient of expected return w.r.t. policy parameters | Tells us how to adjust $\theta$ to increase expected return |
| $\nabla_\theta \log \pi_\theta(a|s)$ | Score function — gradient of log-probability of action $a$ | Direction in parameter space that increases $a$'s probability |
| $Q^{\pi_\theta}(s, a)$ | Action-value function under current policy | How good action $a$ actually was — scales the gradient |

**Why the environment disappears:** The environment dynamics $P(s' \mid s, a)$ do not depend on $\theta$, so they differentiate to zero. This is the "log-derivative trick."

### REINFORCE Algorithm

REINFORCE replaces the true $Q^{\pi_\theta}(s, a)$ with the Monte Carlo return $G_t$ (the actual cumulative reward observed from that point forward).

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]
$$

### Complete Algorithm Steps

1. Initialize policy network $\pi_\theta$ with random weights
2. For each episode:
   a. Generate trajectory: $(s_0, a_0, r_0), (s_1, a_1, r_1), \dots, (s_T, a_T, r_T)$
   b. For each time step $t$, compute discounted return $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$
   c. Compute loss: $\mathcal{L} = -\sum_t \log \pi_\theta(a_t|s_t) \cdot G_t$
   d. Update: $\theta \leftarrow \theta + \alpha \nabla_\theta \mathcal{L}$

### Policy Gradient for Softmax Policy

For a discrete action space with softmax policy, the gradient of log-probability for the chosen action $a$ is:

$$
\nabla_\theta \log \pi_\theta(a|s) = \mathbf{1}_a - \pi_\theta(\cdot|s)
$$

where $\mathbf{1}_a$ is a one-hot vector at action $a$ and $\pi_\theta(\cdot|s)$ is the full probability vector.

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

## 5. Advantage Function and Actor-Critic

### Motivation and Intuition

The returns $G_t$ in REINFORCE have high variance — the same state-action pair can lead to very different returns depending on later randomness. A **baseline** $b(s)$ subtracts a state-dependent value from $G_t$ without changing the expected gradient, reducing variance.

### Advantage Function

$$
A^{\pi_\theta}(s, a) = Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $V^{\pi_\theta}(s)$ | State-value function: expected return from state $s$ | "How good is this state on average?" |
| $A^{\pi_\theta}(s, a)$ | Advantage: how much better action $a$ is than average | Positive = better than average, negative = worse |

**Intuition:** An action might get a return of +50, but if the baseline is +50 (the state is already great), the advantage is 0 — the action was not special. Another action gets +10, but the baseline is -30, so the advantage is +40 — that action was much better than average and should be reinforced.

### Actor-Critic Methods

Actor-critic methods use two neural networks:

- **Actor** (policy network $\pi_\theta$): decides which actions to take
- **Critic** (value network $V_\phi$): estimates the value function to provide a baseline

The critic learns $V_\phi(s)$ by minimizing the mean squared error between its prediction and the actual return:

$$
\mathcal{L}_{critic} = \mathbb{E} \left[ \left( G_t - V_\phi(s_t) \right)^2 \right]
$$

The actor uses the advantage estimated by the critic:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot \left( G_t - V_\phi(s) \right) \right]
$$

Alternatively, the **TD error** $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ can be used as an advantage estimate, enabling single-step updates without waiting for episode completion.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| Actor $\pi_\theta$ | Policy network — outputs action probabilities | What we are optimizing |
| Critic $V_\phi$ | Value network — estimates state value | Provides baseline to reduce policy gradient variance |
| TD error | $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ | Advantage estimate using bootstrapping |

---

## 6. RLHF Pipeline — SFT, Reward Model, PPO

### Motivation and Intuition

Large language models (LLMs) trained on next-token prediction learn statistical patterns in text, but they do not inherently produce outputs that are helpful, harmless, or honest. RLHF steers the model toward human-preferred responses by first learning a reward model from human comparisons, then using reinforcement learning to maximize that reward while staying close to the original model.

### Three Stages of RLHF

| Stage | Description | Output |
| :--- | :--- | :--- |
| 1. Supervised Fine-Tuning (SFT) | Fine-tune pretrained LLM on human-written prompt-response pairs | $\pi_{\text{SFT}}$ |
| 2. Reward Model Training | Train a reward model $R_\phi$ from human preferences (A vs B) | $R_\phi(x, y)$ |
| 3. PPO with KL Penalty | Fine-tune $\pi_\theta$ to maximize reward while staying close to $\pi_{\text{SFT}}$ | $\pi_\theta$ (aligned model) |

### Stage 1: Supervised Fine-Tuning (SFT)

Teach the model to respond in a conversational format via standard supervised learning:

$$
\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{SFT}}} \left[ \sum_{t=1}^{|y|} \log \pi_{\text{SFT}}(y_t \mid x, y_{<t}) \right]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $x$ | Input prompt | Conditions the model on the user's request |
| $y$ | Human-written response | Provides the target distribution to imitate |
| $\pi_{\text{SFT}}$ | Policy (model) after SFT | Starting point for reward modeling and RLHF |

### Stage 2: Reward Model Training

The reward model $R_\phi(x, y)$ takes a response and outputs a scalar score. Humans are shown two responses to the same prompt and pick the better one. Using the Bradley-Terry preference model:

$$
\mathcal{L}_R = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[ \log \sigma \big( R_\phi(x, y_w) - R_\phi(x, y_l) \big) \right]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $y_w$ | Preferred (winning) response | Human judged this response better |
| $y_l$ | Dispreferred (losing) response | Human judged this response worse |
| $R_\phi(x, y)$ | Scalar reward for response $y$ given prompt $x$ | Proxy for human preference |
| $\sigma$ | Sigmoid function | Squashes reward difference into (0, 1) |
| $\mathcal{D}_{\text{pref}}$ | Dataset of human comparisons | Usually thousands to millions of pairwise labels |

### Stage 3: Proximal Policy Optimization with KL Penalty

Fine-tune the SFT model to maximize the reward model's score. A KL penalty prevents the policy from drifting too far from $\pi_{\text{SFT}}$ (which would lead to reward hacking — outputs that score high but are nonsensical).

Full PPO objective for RLHF:

$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}_{(x, y) \sim \pi_\theta} \left[ R_\phi(x, y) \right] - \beta \cdot \mathbb{E}_{x \sim \mathcal{D}} \left[ D_{\text{KL}} \big( \pi_\theta(\cdot \mid x) \;\|\; \pi_{\text{SFT}}(\cdot \mid x) \big) \right]
$$

The PPO clipped surrogate objective (Schulman et al., 2017):

$$
\mathcal{L}^{\text{PPO-clip}} = \mathbb{E} \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\pi_\theta$ | Policy being optimized | The model we are aligning |
| $\pi_{\text{SFT}}$ | Frozen SFT model | Reference distribution to stay near |
| $\beta$ | KL penalty coefficient | Controls how tightly we constrain the policy |
| $D_{\text{KL}}(P \parallel Q)$ | Kullback-Leibler divergence | Measures how much $P$ diverges from $Q$ |
| $r_t(\theta)$ | Importance sampling ratio $\frac{\pi_\theta(y_t|x, y_{<t})}{\pi_{\text{SFT}}(y_t|x, y_{<t})}$ | Corrects for off-policy sampling |
| $\epsilon$ | Clipping hyperparameter | Stabilizes PPO (typically 0.2) |
| $\hat{A}_t$ | Estimated advantage | How much better this action is than average |

**Why RLHF aligns LLMs:**

| Factor | Explanation |
| :--- | :--- |
| Human preferences | Reward model encodes nuanced human values hard to specify in a loss function |
| KL constraint | Prevents model from exploiting the reward model or forgetting its language abilities |
| PPO stability | Clipping and advantage normalization make policy gradient training stable for large models |
| Iterative process | Reward model can be updated with new preferences, enabling continuous improvement |

---

## 7. Code Example — REINFORCE on CartPole

```python
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

    def get_action(self, state):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns G_t."""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns)


def train_reinforce(env_name='CartPole-v1', num_episodes=500, gamma=0.99, lr=1e-3):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNet(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        states, actions, rewards = [], [], []

        while True:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob = policy.get_action(state_t)
            next_state, reward, done, truncated, _ = env.step(action)

            states.append(state)
            actions.append(log_prob)
            rewards.append(reward)

            state = next_state
            if done or truncated:
                break

        # Compute returns and update policy
        returns = compute_returns(rewards, gamma)
        policy_loss = []
        for log_prob, G_t in zip(actions, returns):
            policy_loss.append(-log_prob * G_t)

        optimizer.zero_grad()
        torch.stack(policy_loss).sum().backward()
        optimizer.step()

        episode_rewards.append(sum(rewards))
        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}, Avg Reward (last 100): {avg:.1f}")

    env.close()
    return episode_rewards
```

---

## 8. Exploration vs Exploitation — Pros and Cons

### The Exploration-Exploitation Dilemma

| Strategy | Description | Trade-off |
| :--- | :--- | :--- |
| Greedy | Always take best-known action | Fast convergence to suboptimal policy (may miss better options) |
| Epsilon-greedy | With probability $\epsilon$, take random action | Simple, but explores uniformly (no memory) |
| Softmax exploration | Sample proportionally to action preferences | Natural in policy gradients, but slow to focus |
| Upper Confidence Bound (UCB) | Choose action with highest potential upside | Optimistic exploration, more systematic |
| Thompson sampling | Sample from posterior over Q-values | Bayes-optimal, but expensive |

**Exploration in policy gradients:** The softmax output naturally provides exploration — even when one action is strongly preferred, others have nonzero probability. **Temperature** controls this balance:

$$
\pi_\theta(a|s) = \frac{e^{z_a / \tau}}{\sum_{a'} e^{z_{a'} / \tau}}
$$

| $\tau$ | Effect | Use Case |
| :--- | :--- | :--- |
| High ($>1$) | More uniform — more exploration | Early training, unknown environments |
| Low ($<1$) | Sharper — more exploitation | Later training, fine-tuning |
| $\tau \to 0$ | Greedy — always pick most likely action | Deployment, deterministic inference |

### Pros and Cons of RL Approaches

| Approach | Pros | Cons |
| :--- | :--- | :--- |
| **Q-Learning / DQN** | Off-policy (reuses past experience), sample efficient | Overestimates Q-values, unstable with high-dimensional actions |
| **Policy Gradients (REINFORCE)** | Handles continuous action spaces, stochastic policies guaranteed convergence | High variance, sample inefficient |
| **Actor-Critic** | Lower variance than REINFORCE, can be on or off-policy | Two networks to train, more hyperparameters |
| **PPO** | Stable, reliable, widely used | More complex, requires careful clipping tuning |
| **RLHF** | Aligns LLMs with human preferences | Expensive human data collection, reward model bias |

### When to Use What

| Scenario | Recommendation |
| :--- | :--- |
| Discrete actions, small state space | Q-learning (tabular) |
| High-dimensional state (images) | DQN or DQN variants |
| Continuous control (robotics) | PPO or SAC |
| Need guaranteed convergence | Policy gradients |
| LLM alignment | RLHF (PPO + reward model) |
| Sample efficiency critical | Off-policy methods (DQN, SAC) |

**ML Connection:** RL is the foundation for LLM alignment (RLHF/PPO for ChatGPT, Claude, Gemini), game AI (AlphaGo, AlphaZero), robotics (grasping, locomotion), autonomous driving (motion planning), and recommendation systems (dynamic user modeling).

> **Check your intuition:** In REINFORCE, suppose an action leads to a very positive return $G_t = 100$, but the baseline $b(s) = 95$. The advantage is only 5. Another action gets $G_t = 20$ with baseline $b(s) = -10$, so the advantage is 30. Which action gets reinforced more strongly? Why does the baseline make this correct? — The second action is reinforced more because it was much better than expected given the poor state, while the first action was only slightly better than expected in an already good state.

---

## Prerequisites and Further Reading

- **StatQuest:** Reinforcement Learning with Neural Networks (L23), RL Mathematical Details (L24), RLHF (L25), Decoder-Only Transformers (L21)
- **Textbook:** Sutton & Barto, "Reinforcement Learning: An Introduction" — the definitive reference
- **DQN:** Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013)
- **PPO:** Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **RLHF:** Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022)
- **Concepts:** Policy gradients, cross-entropy loss, KL divergence, transformer architectures, softmax function, discounting
