# solvers/actor_critic.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base_solver import BaseSolver


class ActorCriticNetwork(nn.Module):
    """
    Shared network for both actor (policy) and critic (value function).
    """
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


class ActorCriticSolver(BaseSolver):
    """
    Advantage Actor-Critic (A2C) solver for Appalachian Trail.
    """

    def __init__(self, env, lr=1e-3, gamma=0.99, hidden_size=128, seed=None):
        super().__init__(env, seed)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        self.gamma = gamma

        self.network = ActorCriticNetwork(obs_dim, act_dim, hidden_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.log_probs = []
        self.values = []
        self.rewards = []

    # -----------------------------------------------------
    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        probs, value = self.network(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))
        self.values.append(value)
        return int(action.item())

    # -----------------------------------------------------
    def _compute_returns_and_advantages(self):
        """
        Compute discounted returns and advantages for A2C.
        """
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.cat(self.values).squeeze()
        advantages = returns - values

        return returns, advantages

    # -----------------------------------------------------
    def _update(self):
        returns, advantages = self._compute_returns_and_advantages()

        actor_loss = -torch.stack(self.log_probs) * advantages.detach()
        actor_loss = actor_loss.mean()

        critic_loss = advantages.pow(2).mean()

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear buffers
        self.log_probs = []
        self.values = []
        self.rewards = []

    # -----------------------------------------------------
    def train(self, episodes=500):
        episode_rewards = []

        for ep in range(episodes):
            state, _ = self.env.reset()
            done = False
            ep_reward = 0

            while not done:
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                self.rewards.append(reward)
                ep_reward += reward
                state = next_state
                done = terminated or truncated

            # Update network at the end of episode
            self._update()
            episode_rewards.append(ep_reward)

            if (ep + 1) % 50 == 0:
                print(f"Episode {ep+1}/{episodes} | Reward: {ep_reward:.1f}")

        return episode_rewards
