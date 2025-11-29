# solvers/base_solver.py
from abc import ABC, abstractmethod
import numpy as np

class BaseSolver(ABC):
    """
    Abstract base class for all RL solvers.

    All solvers must implement:
    - act(state): choose an action
    - train(episodes): run training loop

    This class provides:
    - env reference
    - RNG
    - run_episode() helper for consistency
    """
    def __init__(self, env, seed=None):
        self.env = env
        self.rng = np.random.default_rng(seed)
    
     # -----------------------------------------------------
    @abstractmethod
    def act(self, state):
        """Return an action given the current state."""
        pass

    # -----------------------------------------------------
    @abstractmethod
    def train(self, episodes: int):
        """Train the agent for a number of episodes."""
        pass
    
    # -----------------------------------------------------
    def run_episode(self, render=False):
        """
        Utility helper: run a single episode using self.act().
        Useful for evaluation or debugging.
        """

        state, _ = self.env.reset()
        total_reward = 0
        done = False

        while not done:
            action = self.act(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)

            total_reward += reward
            state = next_state

            done = terminated or truncated

            if render:
                self.env.render()

        return total_reward
