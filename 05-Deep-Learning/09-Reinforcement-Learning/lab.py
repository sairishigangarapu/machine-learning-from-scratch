import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =========================================================================
# Policy Network
# =========================================================================

class PolicyNet(nn.Module):
    """Simple feedforward policy network for discrete action spaces."""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

    def get_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), probs.squeeze(0).detach().numpy()


# =========================================================================
# REINFORCE Implementation
# =========================================================================

def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns G_t for a complete episode."""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def reinforce(env_name='CartPole-v1', num_episodes=500, gamma=0.99, lr=1e-3):
    """Train a policy using the REINFORCE (Monte Carlo policy gradient) algorithm."""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Gamma (discount): {gamma}")
    print(f"Learning rate: {lr}")

    policy = PolicyNet(state_dim, action_dim, hidden_dim=128)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_rewards = []
    episode_lengths = []
    best_avg_reward = -float('inf')

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        log_probs = []
        rewards = []

        while True:
            action, log_prob, _ = policy.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state
            if done or truncated:
                break

        # Compute discounted returns
        returns = compute_returns(rewards, gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns (reduces variance)
        if len(returns) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-9)

        # Policy gradient loss: -sum(log_prob * G_t)
        policy_loss = []
        for log_prob, G_t in zip(log_probs, returns_t):
            policy_loss.append(-log_prob * G_t)

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        episode_rewards.append(sum(rewards))
        episode_lengths.append(len(rewards))

        # Track and report progress
        if episode >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
        else:
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward

        if episode % 50 == 0:
            print(f"Episode {episode:4d}/{num_episodes} | "
                  f"Reward: {sum(rewards):5.1f} | "
                  f"Avg(100): {avg_reward:5.1f} | "
                  f"Avg Len: {avg_length:4.1f}")

    env.close()
    return episode_rewards, episode_lengths


# =========================================================================
# Value Network (Critic) for Actor-Critic
# =========================================================================

class ActorCritic(nn.Module):
    """Shared network with actor (policy) and critic (value) heads."""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Linear(state_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        action_logits = self.actor(x)
        value = self.critic(x)
        return F.softmax(action_logits, dim=-1), value

    def get_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs, value = self.forward(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(0), probs.squeeze(0).detach().numpy()


def actor_critic(env_name='CartPole-v1', num_episodes=500, gamma=0.99, lr=1e-3):
    """Train using Advantage Actor-Critic (A2C)."""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim, hidden_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    episode_rewards = []

    print(f"\nTraining Actor-Critic on {env_name}...")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        log_probs = []
        values = []
        rewards = []

        while True:
            action, log_prob, value, _ = model.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state
            if done or truncated:
                break

        # Compute TD targets and advantages
        returns = compute_returns(rewards, gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.stack(values).squeeze()
        log_probs_t = torch.stack(log_probs)

        # Normalize returns
        if len(returns) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-9)

        # Advantage = G_t - V(s)
        advantages = returns_t - values_t.detach()

        # Actor loss: -sum(log_prob * advantage)
        actor_loss = -(log_probs_t * advantages).sum()

        # Critic loss: MSE between predicted value and actual return
        critic_loss = F.mse_loss(values_t, returns_t)

        # Combined loss
        total_loss = actor_loss + critic_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        episode_rewards.append(sum(rewards))

        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode:4d}/{num_episodes} | Avg Reward (100): {avg_reward:5.1f}")

    env.close()
    return episode_rewards


# =========================================================================
# Visualization
# =========================================================================

def plot_rewards(rewards, title="Training Rewards", save_path="rl_rewards.png", window=50):
    """Plot episode rewards with a smoothed running average."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(rewards, alpha=0.3, color='blue', label='Episode Reward')

    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window - 1, len(rewards)), smoothed, color='red',
                linewidth=2, label=f'{window}-episode Moving Avg')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"Reward plot saved to {save_path}")


def test_trained_policy(model, env_name='CartPole-v1', num_episodes=10, render=False):
    """Evaluate a trained policy on multiple episodes."""
    env = gym.make(env_name, render_mode='human' if render else None)
    rewards = []

    print(f"\nTesting trained policy ({num_episodes} episodes):")
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        while True:
            action, _, _, _ = model.get_action(state)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if done or truncated:
                break
        rewards.append(total_reward)

    env.close()
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"  Mean reward: {mean_reward:.1f} +/- {std_reward:.1f}")
    print(f"  Min: {min(rewards)}, Max: {max(rewards)}")
    return rewards


# =========================================================================
# Simple grid world for visualizing policy
# =========================================================================

class SimpleGridWorld:
    """A 4x4 grid world with a goal state for RL demonstration."""
    def __init__(self, size=4):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.agent_pos = self.start
        self.actions = ['up', 'down', 'left', 'right']

    def reset(self):
        self.agent_pos = self.start
        return self._get_state()

    def _get_state(self):
        """Return (row, col) normalized to [0, 1]."""
        return np.array([self.agent_pos[0] / (self.size - 1),
                         self.agent_pos[1] / (self.size - 1)])

    def step(self, action_idx):
        r, c = self.agent_pos
        if action_idx == 0:   # up
            r = max(0, r - 1)
        elif action_idx == 1: # down
            r = min(self.size - 1, r + 1)
        elif action_idx == 2: # left
            c = max(0, c - 1)
        elif action_idx == 3: # right
            c = min(self.size - 1, c + 1)

        self.agent_pos = (r, c)
        done = (self.agent_pos == self.goal)
        reward = 1.0 if done else -0.1
        return self._get_state(), reward, done, False, {}


def train_gridworld():
    """Train REINFORCE on the simple grid world and visualize the learned policy."""
    print("\n" + "=" * 65)
    print("BONUS: Simple Grid World — Policy Visualization")
    print("=" * 65)

    env = SimpleGridWorld(size=4)
    policy = PolicyNet(state_dim=2, action_dim=4, hidden_dim=32)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    action_names = ['\u2191', '\u2193', '\u2190', '\u2192']  # arrows

    for episode in range(200):
        state = env.reset()
        log_probs = []
        rewards = []

        while True:
            action, log_prob, _ = policy.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            if done:
                break

        returns = compute_returns(rewards, gamma=0.9)
        policy_loss = [-lp * G for lp, G in zip(log_probs, returns)]

        optimizer.zero_grad()
        torch.stack(policy_loss).sum().backward()
        optimizer.step()

    # Visualize learned policy
    print("Learned greedy policy (arrows show preferred action per cell):")
    print("Goal = bottom-right\n")

    grid = [['' for _ in range(4)] for _ in range(4)]
    for r in range(4):
        for c in range(4):
            if (r, c) == env.goal:
                grid[r][c] = 'G'
            else:
                state = np.array([r / 3.0, c / 3.0])
                with torch.no_grad():
                    probs = policy(torch.FloatTensor(state).unsqueeze(0)).squeeze().numpy()
                best_action = np.argmax(probs)
                confidence = probs[best_action]
                grid[r][c] = f"{action_names[best_action]} ({confidence:.2f})"

    print("     Col 0        Col 1        Col 2        Col 3")
    for r, row in enumerate(grid):
        print(f"Row {r}: ", "  ".join(f"{cell:<12}" for cell in row))

    return policy


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 65)
    print("Lab: Reinforcement Learning — REINFORCE & Actor-Critic")
    print("=" * 65)

    # -----------------------------------------------------------------
    # PART 1: Scalar check — discounted returns
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 1: Discounted Returns — Manual Computation")
    print("-" * 65)

    rewards_example = [1, 1, 1, 1, 10]
    gamma = 0.9
    returns = compute_returns(rewards_example, gamma)
    print(f"Rewards: {rewards_example}")
    print(f"Gamma: {gamma}")
    print(f"Returns G_t: {[f'{g:.2f}' for g in returns]}")
    print(f"  G_0 = 1 + 0.9*1 + 0.9^2*1 + 0.9^3*1 + 0.9^4*10")
    print(f"  G_0 = {returns[0]:.2f}")

    # -----------------------------------------------------------------
    # PART 2: Policy Network Forward Pass
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 2: Policy Network — Forward Pass Demo")
    print("-" * 65)

    torch.manual_seed(42)
    state_dim = 4
    action_dim = 2
    policy_demo = PolicyNet(state_dim, action_dim, hidden_dim=16)

    sample_state = np.array([0.1, -0.2, 0.3, 0.0])
    probs = policy_demo(torch.FloatTensor(sample_state).unsqueeze(0)).squeeze().detach().numpy()
    print(f"State: {sample_state}")
    print(f"Action probabilities: Left={probs[0]:.4f}, Right={probs[1]:.4f}")
    print(f"Sum: {probs.sum():.4f}")

    action, log_prob, probs_sampled = policy_demo.get_action(sample_state)
    print(f"Sampled action: {action}, Log prob: {log_prob.item():.4f}")

    # -----------------------------------------------------------------
    # PART 3: Train REINFORCE on CartPole
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 3: REINFORCE on CartPole-v1")
    print("-" * 65)

    reinforce_rewards, reinforce_lengths = reinforce(
        env_name='CartPole-v1',
        num_episodes=300,
        gamma=0.99,
        lr=1e-3
    )

    print(f"\nREINFORCE results:")
    print(f"  Best avg reward (100 ep): {max(np.convolve(reinforce_rewards, np.ones(100)/100, mode='valid')):.1f}")
    print(f"  Final avg reward (last 100): {np.mean(reinforce_rewards[-100:]):.1f}")

    plot_rewards(reinforce_rewards, "REINFORCE on CartPole", "rl_reinforce_rewards.png")

    # -----------------------------------------------------------------
    # PART 4: Train Actor-Critic on CartPole
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 4: Actor-Critic (A2C) on CartPole-v1")
    print("-" * 65)

    ac_rewards = actor_critic(
        env_name='CartPole-v1',
        num_episodes=300,
        gamma=0.99,
        lr=1e-3
    )

    print(f"\nActor-Critic results:")
    print(f"  Best avg reward (100 ep): {max(np.convolve(ac_rewards, np.ones(100)/100, mode='valid')):.1f}")
    print(f"  Final avg reward (last 100): {np.mean(ac_rewards[-100:]):.1f}")

    plot_rewards(ac_rewards, "Actor-Critic on CartPole", "rl_actor_critic_rewards.png")

    # -----------------------------------------------------------------
    # PART 5: Test Trained Policy (REINFORCE)
    # -----------------------------------------------------------------
    print("\n" + "-" * 65)
    print("PART 5: Testing Trained Policy")
    print("-" * 65)

    # Re-train briefly and test
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    test_policy = PolicyNet(state_dim, action_dim, hidden_dim=128)
    optimizer = optim.Adam(test_policy.parameters(), lr=1e-3)

    for ep in range(200):
        state, _ = gym.make('CartPole-v1').reset()
        log_probs, rewards = [], []
        env_ep = gym.make('CartPole-v1')
        state, _ = env_ep.reset()
        while True:
            action, log_prob, _ = test_policy.get_action(state)
            next_state, reward, done, truncated, _ = env_ep.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            if done or truncated:
                break
        env_ep.close()

        returns = compute_returns(rewards, gamma=0.99)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-9)
        loss = torch.stack([-lp * G for lp, G in zip(log_probs, returns_t)]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_rewards = []
    env_test = gym.make('CartPole-v1')
    for ep in range(20):
        state, _ = env_test.reset()
        ep_rew = 0
        while True:
            action, _, _ = test_policy.get_action(state)
            state, reward, done, truncated, _ = env_test.step(action)
            ep_rew += reward
            if done or truncated:
                break
        test_rewards.append(ep_rew)
    env_test.close()

    print(f"Test over 20 episodes:")
    print(f"  Mean reward: {np.mean(test_rewards):.1f}")
    print(f"  Max reward: {max(test_rewards)}")
    print(f"  Solved (>=195.0 episodes): {sum(r >= 195.0 for r in test_rewards)}/20")

    # -----------------------------------------------------------------
    # PART 6: Grid World Policy Visualization
    # -----------------------------------------------------------------
    train_gridworld()

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Lab complete! Key takeaways:")
    print("  1. REINFORCE collects full episodes and updates using Monte Carlo returns")
    print("  2. The policy gradient uses the log-derivative trick to differentiate through stochastic actions")
    print("  3. Actor-Critic adds a value network (critic) to reduce variance via a baseline")
    print("  4. Return normalization improves training stability")
    print("  5. The learned policy can be visualized as a heatmap of action preferences")
    print("=" * 65)


if __name__ == "__main__":
    main()
