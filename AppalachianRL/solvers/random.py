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
        failure_reasons = []

        for ep in range(episodes):
            state, _ = self.env.reset()
            done = False
            total_r = 0
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_r += reward
                state = next_state

                # if episode ended, record reason
                if done:
                    failure_reasons.append(info.get('failure_reason', 'Unknown'))
            
            rewards.append(total_r)
            
            if (ep + 1) % 50 == 0:
                print(f"Episode {ep+1}/{episodes} | Reward: {total_r:.1f}")

        return rewards, failure_reasons