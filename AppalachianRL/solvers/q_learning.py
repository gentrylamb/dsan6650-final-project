# solvers/q_learning.py
import numpy as np
from .base_solver import BaseSolver


class QLearningSolver(BaseSolver):
    """
    Tabular Q-learning agent for the Appalachian Trail environment.
    Continuous state variables are discretized into bins.
    """

    def __init__(
        self,
        env,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=0.1,
        bins=(40, 20, 20, 3, 200), 
        # bins=(10, 5, 3, 10),        # for env without energy
        epsilon_decay=None,         # optional
        seed=None
    ):
        super().__init__(env, seed)

        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.bins = bins
        self.bin_edges = self._create_bin_edges()

        # Q-table shape: (#bins_0, ..., #bins_4, #actions)
        self.q_table = np.zeros((*bins, env.action_space.n))

    # -----------------------------------------------------
    # State Discretization
    # -----------------------------------------------------
    def _create_bin_edges(self):
        lows = self.env.observation_space.low
        highs = self.env.observation_space.high

        edges = []
        for low, high, num in zip(lows, highs, self.bins):
            # Ensure proper scaling for large ranges
            edges.append(np.linspace(low, high, num + 1)[1:-1])
        return edges

    def _discretize(self, state):
        # Clip state to avoid OOB values
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        state = np.clip(state, low, high)

        return tuple(
            np.digitize(s, edges)
            for s, edges in zip(state, self.bin_edges)
        )

    # -----------------------------------------------------
    # Action Selection (ε-greedy)
    # -----------------------------------------------------
    def act(self, state):
        if self.rng.random() < self.epsilon:
            return self.env.action_space.sample()

        s = self._discretize(state)
        return int(np.argmax(self.q_table[s]))

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    def train(self, episodes=5000):
        rewards = []
        failure_reasons = []

        for ep in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_r = 0

            while not done:
                s_disc = self._discretize(state)

                # ε-greedy action
                if self.rng.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = int(np.argmax(self.q_table[s_disc]))

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                s_next_disc = self._discretize(next_state)

                # Normalize rewards mildly to stabilize Q updates
                norm_reward = reward / 50.0

                # Terminal state handling
                if done:
                    td_target = norm_reward
                    failure_reasons.append(info.get("failure_reason", "Unknown"))
                else:
                    td_target = norm_reward + self.gamma * np.max(self.q_table[s_next_disc])

                td_error = td_target - self.q_table[s_disc][action]
                self.q_table[s_disc][action] += self.lr * td_error

                state = next_state
                total_r += reward

            # Optional epsilon decay
            if self.epsilon_decay is not None:
                self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

            rewards.append(total_r)

            if (ep + 1) % 50 == 0:
                print(f"Episode {ep+1}/{episodes} | Reward: {total_r:.1f}")

        return rewards, failure_reasons
