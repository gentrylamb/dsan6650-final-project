# tests/test_env.py

import numpy as np
from AppalachianRL.envs.trail import AppalachianTrailEnv


def test_env_reset():
    env = AppalachianTrailEnv()
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert len(obs) == 5  # miles, energy, food, weather, day


def test_env_step():
    env = AppalachianTrailEnv()
    obs, _ = env.reset()

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == (5,)
    assert isinstance(reward, (float, int))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_env_run_full_episode():
    env = AppalachianTrailEnv()
    obs, _ = env.reset()

    for _ in range(1000):  # should finish long before this
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    assert terminated or truncated
