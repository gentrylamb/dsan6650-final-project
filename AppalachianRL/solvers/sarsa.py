# solvers/sarsa.py
import numpy as np
from .base_solver import BaseSolver


class SarsaSolver(BaseSolver):
    """
    Tabular SARSA (on-policy TD) solver for the Appalachian Trail environment.
    Continuous state variables are discretized into bins.
    """

    def __init__(
        self,
        env,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=0.1,
        bins=(40, 20, 20, 3, 200),
        # bins=(10, 5, 3, 10),         # for env without energy
        epsilon_decay=None,          # optional
        seed=None
    ):
        super().__init__(env, seed)

        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

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
        # Clip state to avoid going outside bin edges
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        state = np.clip(state, low, high)

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
        failure_reasons = []

        for ep in range(episodes):
            state, info = self.env.reset()
            s_disc = self._discretize(state)

            # Choose initial action
            if self.rng.random() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                action = int(np.argmax(self.q_table[s_disc]))

            total_r = 0
            done = False

            while not done:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_r += reward

                s_next_disc = self._discretize(next_state)

                # Normalize reward to stabilize learning
                # reward = reward / 50.0

                # Choose next action (on-policy)
                if not done:
                    if self.rng.random() < self.epsilon:
                        next_action = self.env.action_space.sample()
                    else:
                        next_action = int(np.argmax(self.q_table[s_next_disc]))
                else:
                    next_action = None  # terminal

                # SARSA target
                if done:
                    td_target = reward
                    failure_reasons.append(info.get("failure_reason", "Unknown"))
                else:
                    td_target = reward + self.gamma * self.q_table[s_next_disc][next_action]

                td_error = td_target - self.q_table[s_disc][action]

                # Update
                self.q_table[s_disc][action] += self.lr * td_error

                # Advance
                state = next_state
                s_disc = s_next_disc
                action = next_action

            # Optional epsilon decay
            if self.epsilon_decay is not None:
                self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

            rewards.append(total_r)

            if (ep + 1) % 50 == 0:
                print(f"Episode {ep+1}/{episodes} | Reward: {total_r:.1f}")

        return rewards, failure_reasons


