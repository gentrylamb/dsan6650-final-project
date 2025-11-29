# solvers/sarsa.py
import numpy as np
from .base_solver import BaseSolver


class SarsaSolver(BaseSolver):
    """
    Tabular SARSA (on-policy TD) solver for the Appalachian Trail environment.
    Uses the same state discretization as the Q-learning agent.
    """

    def __init__(
        self,
        env,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=0.1,
        bins=(20, 10, 10, 3, 50),
        seed=None
    ):
        super().__init__(env, seed)

        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # Discretization setup
        self.bins = bins
        self.bin_edges = self._create_bin_edges()

        # Q-table
        self.q_table = np.zeros((*bins, env.action_space.n))

    # -----------------------------------------------------
    # Discretization helpers
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
    def act(self, state):
        """ε-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_table[self._discretize(state)]))

    # -----------------------------------------------------
    def train(self, episodes=5000):
        rewards = []

        for ep in range(episodes):
            state, _ = self.env.reset()
            s_disc = self._discretize(state)

            # Choose initial action
            if self.rng.random() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                action = int(np.argmax(self.q_table[s_disc]))

            total_r = 0
            done = False

            while not done:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_r += reward

                s_next_disc = self._discretize(next_state)

                # Choose next action (on-policy)
                if self.rng.random() < self.epsilon:
                    next_action = self.env.action_space.sample()
                else:
                    next_action = int(np.argmax(self.q_table[s_next_disc]))

                # SARSA target
                td_target = reward + self.gamma * self.q_table[s_next_disc][next_action] * (not done)
                td_error = td_target - self.q_table[s_disc][action]

                # Update
                self.q_table[s_disc][action] += self.lr * td_error

                # Advance
                state = next_state
                s_disc = s_next_disc
                action = next_action

            rewards.append(total_r)

        return rewards
