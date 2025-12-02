# solvers/policy_gradient.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_solver import BaseSolver


class PolicyNetwork(nn.Module):
    """
    Simple MLP policy π(a|s) producing action probabilities.
    """
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)


class PolicyGradientSolver(BaseSolver):
    """
    Monte Carlo Policy Gradient implementation.
    """

    def __init__(
        self,
        env,
        lr=1e-3,
        gamma=0.99,
        hidden_size=128,
        seed=None
    ):
        super().__init__(env, seed)

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        self.gamma = gamma

        # Create policy network
        self.policy = PolicyNetwork(obs_dim, act_dim, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Store trajectories
        self.log_probs = []
        self.rewards = []

    # -----------------------------------------------------
    def act(self, state):
        """
        Sample an action from π(a|s).
        """

        state_tensor = torch.tensor(state, dtype=torch.float32)
        probs = self.policy(state_tensor)

        # Categorical sampling
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        # store log-prob for training
        log_prob = dist.log_prob(action)
        self.log_probs.append(log_prob)

        return int(action.item())

    # -----------------------------------------------------
    def _compute_returns(self):
        """
        Compute discounted returns G_t for each step:
        G_t = r_t + gamam*r_{t+1} + gamma^2*r_{t+2} + ...
        """
        returns = []
        G = 0

        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.append(G)

        returns.reverse()
        return returns

    # -----------------------------------------------------
    def _update_policy(self):
        """
        Apply REINFORCE gradient update:
        ∇J = E[ G_t ∇ log π(a_t|s_t) ]
        """
        returns = self._compute_returns()

        # Normalize returns for training stability
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, Gt in zip(self.log_probs, returns):
            loss -= log_prob * Gt   # gradient ascent via negative loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear buffers
        self.log_probs.clear()
        self.rewards.clear()

    # -----------------------------------------------------
    def train(self, episodes=500):
        """
        Train using full-episode rollouts with REINFORCE.
        """

        episode_rewards = []
        failure_reasons = []

        for ep in range(episodes):
            state, _ = self.env.reset()
            done = False
            ep_reward = 0

            while not done:
                action = self.act(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # Normalize reward
                # reward = reward / 50

                self.rewards.append(reward)
                ep_reward += reward

                state = next_state
                done = terminated or truncated

                # if episode ended, record reason
                if done:
                    failure_reasons.append(info.get('failure_reason', 'Unknown'))

            # Episode finished → update policy
            self._update_policy()

            episode_rewards.append(ep_reward)

            if (ep + 1) % 50 == 0:
                print(f"Episode {ep+1}/{episodes} | Reward: {ep_reward:.1f}")

        return episode_rewards, failure_reasons