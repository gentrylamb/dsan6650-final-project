# solvers/random.py
import numpy as np


class RandomAgent():
    """
    Agent that makes random acts as a baseline to compare.
    """
    def __init__(self, env, seed=None):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def act():
        pass

    def train(self, episodes=1000):
        rewards = []
        for ep in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_r = 0
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_r += reward
                state = next_state
            rewards.append(total_r)
            
        return rewards