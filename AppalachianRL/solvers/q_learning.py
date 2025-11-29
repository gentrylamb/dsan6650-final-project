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
        bins=(20, 10, 10, 3, 50),  # discretization for each state dimension
        seed=None
    ):
        super().__init__(env, seed)

        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # Discretization bins (for: miles, energy, food, weather, day)
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
            edges.append(np.linspace(low, high, num + 1)[1:-1])
        return edges

    def _discretize(self, state):
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

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                s_next_disc = self._discretize(next_state)

                # Q-learning target
                best_next_q = np.max(self.q_table[s_next_disc])
                td_target = reward + self.gamma * best_next_q
                td_error = td_target - self.q_table[s_disc][action]

                # Update rule
                self.q_table[s_disc][action] += self.lr * td_error

                state = next_state
                total_r += reward

            rewards.append(total_r)

        return rewards
