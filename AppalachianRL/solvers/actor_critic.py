# solvers/actor_critic.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base_solver import BaseSolver
from collections import deque
import random

# ---------------------------------------------------------
# Replay buffer for off-policy updates
# ---------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d)

    def __len__(self):
        return len(self.buffer)

# ---------------------------------------------------------
# Actor network (policy)
# ---------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Linear(hidden_size, act_dim)

    def forward(self, x):
        h = self.net(x)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, x):
        mean, std = self(x)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        action = torch.tanh(action)  # bound actions between -1 and 1
        return action, log_prob

# ---------------------------------------------------------
# Critic network (Q-function)
# ---------------------------------------------------------
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

# ---------------------------------------------------------
# Off-policy SAC solver
# ---------------------------------------------------------
class ActorCriticSolver(BaseSolver):
    """
    Soft Actor-Critic (off-policy) for Appalachian Trail.
    Fully off-policy version compatible with the previous interface.
    """
    def __init__(self, env, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 hidden_size=128, batch_size=64, buffer_size=10000, seed=None):
        super().__init__(env, seed)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n  # discrete actions (convert to one-hot)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

        self.actor = Actor(self.obs_dim, self.act_dim, hidden_size)
        self.critic1 = Critic(self.obs_dim, self.act_dim, hidden_size)
        self.critic2 = Critic(self.obs_dim, self.act_dim, hidden_size)
        self.critic1_target = Critic(self.obs_dim, self.act_dim, hidden_size)
        self.critic2_target = Critic(self.obs_dim, self.act_dim, hidden_size)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_size)

    # -----------------------------------------------------
    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs, _ = self.actor.sample(state_tensor)
        action_idx = int(torch.argmax(action_probs).item())
        return action_idx

    # -----------------------------------------------------
    def _update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32)
        actions_onehot = torch.nn.functional.one_hot(torch.tensor(actions), num_classes=self.act_dim).float()
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)

        # -----------------------
        # Update Critic networks
        # -----------------------
        with torch.no_grad():
            next_actions, logp_next = self.actor.sample(next_states)
            q1_target = self.critic1_target(next_states, next_actions)
            q2_target = self.critic2_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * logp_next.unsqueeze(-1)
            y = rewards + self.gamma * (1 - dones) * q_target

        q1 = self.critic1(states, actions_onehot)
        q2 = self.critic2(states, actions_onehot)
        critic1_loss = nn.MSELoss()(q1, y)
        critic2_loss = nn.MSELoss()(q2, y)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        # -----------------------
        # Update Actor network
        # -----------------------
        new_actions, logp = self.actor.sample(states)
        q1_pi = self.critic1(states, new_actions)
        q2_pi = self.critic2(states, new_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp.unsqueeze(-1) - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # -----------------------
        # Soft update targets
        # -----------------------
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # -----------------------------------------------------
    def train(self, episodes=500):
        episode_rewards = []
        failure_reasons = []

        for ep in range(episodes):
            state, info = self.env.reset()
            done = False
            ep_reward = 0

            while not done:
                action = self.act(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Store in replay buffer
                self.buffer.push(state, action, reward, next_state, done)

                ep_reward += reward
                state = next_state

                if done:
                    failure_reasons.append(info.get("failure_reason", "Unknown"))

                # Perform SAC updates at every step
                self._update()

            episode_rewards.append(ep_reward)
            if (ep + 1) % 50 == 0:
                print(f"Episode {ep+1}/{episodes} | Reward: {ep_reward:.1f}")

        return episode_rewards, failure_reasons
















## ON-POLICY A2C VERSION

# # solvers/actor_critic.py
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from .base_solver import BaseSolver


# class ActorCriticNetwork(nn.Module):
#     """
#     Shared network for both actor (policy) and critic (value function).
#     """
#     def __init__(self, input_dim, output_dim, hidden_size=128):
#         super().__init__()
#         self.shared = nn.Sequential(
#             nn.Linear(input_dim, hidden_size),
#             nn.ReLU(),
#         )
#         self.actor = nn.Sequential(
#             nn.Linear(hidden_size, output_dim),
#             nn.Softmax(dim=-1)
#         )
#         self.critic = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         x = self.shared(x)
#         return self.actor(x), self.critic(x)


# class ActorCriticSolver(BaseSolver):
#     """
#     Advantage Actor-Critic (A2C) solver for Appalachian Trail.
#     """

#     def __init__(self, env, lr=1e-3, gamma=0.99, hidden_size=128, seed=None):
#         super().__init__(env, seed)
#         obs_dim = env.observation_space.shape[0]
#         act_dim = env.action_space.n

#         self.gamma = gamma

#         self.network = ActorCriticNetwork(obs_dim, act_dim, hidden_size)
#         self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

#         self.log_probs = []
#         self.values = []
#         self.rewards = []

#     # -----------------------------------------------------
#     def act(self, state):
#         state_tensor = torch.tensor(state, dtype=torch.float32)
#         probs, value = self.network(state_tensor)
#         dist = torch.distributions.Categorical(probs)
#         action = dist.sample()

#         self.log_probs.append(dist.log_prob(action))
#         self.values.append(value)
#         return int(action.item())

#     # -----------------------------------------------------
#     def _compute_returns_and_advantages(self):
#         """
#         Compute discounted returns and advantages for A2C.
#         """
#         returns = []
#         G = 0
#         for r in reversed(self.rewards):
#             G = r + self.gamma * G
#             returns.insert(0, G)

#         returns = torch.tensor(returns, dtype=torch.float32)
#         values = torch.cat(self.values).squeeze()
#         advantages = returns - values

#         return returns, advantages

#     # -----------------------------------------------------
#     def _update(self):
#         returns, advantages = self._compute_returns_and_advantages()

#         actor_loss = -torch.stack(self.log_probs) * advantages.detach()
#         actor_loss = actor_loss.mean()

#         critic_loss = advantages.pow(2).mean()

#         loss = actor_loss + critic_loss

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         # Clear buffers
#         self.log_probs = []
#         self.values = []
#         self.rewards = []

#     # -----------------------------------------------------
#     def train(self, episodes=500):
#         episode_rewards = []
#         failure_reasons = []

#         for ep in range(episodes):
#             state, _ = self.env.reset()
#             done = False
#             ep_reward = 0

#             while not done:
#                 action = self.act(state)
#                 next_state, reward, terminated, truncated, info = self.env.step(action)

#                 self.rewards.append(reward)
#                 ep_reward += reward
#                 state = next_state
#                 done = terminated or truncated

#                 # if episode ended, record reason
#                 if done:
#                     failure_reasons.append(info.get('failure_reason', 'Unknown'))

#             # Update network at the end of episode
#             self._update()
#             episode_rewards.append(ep_reward)

#             if (ep + 1) % 50 == 0:
#                 print(f"Episode {ep+1}/{episodes} | Reward: {ep_reward:.1f}")

#         return episode_rewards, failure_reasons
