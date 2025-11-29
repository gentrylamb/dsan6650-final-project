# tests/test_solvers.py

from AppalachianRL.envs.trail import AppalachianTrailEnv
from AppalachianRL.solvers.q_learning import QLearningSolver
from AppalachianRL.solvers.sarsa import SarsaSolver
from AppalachianRL.solvers.policy_gradient import PolicyGradientSolver


def test_q_learning_runs():
    env = AppalachianTrailEnv()
    agent = QLearningSolver(env)
    rewards = agent.train(episodes=5)

    assert len(rewards) == 5
    assert all(isinstance(r, (float, int)) for r in rewards)


def test_sarsa_runs():
    env = AppalachianTrailEnv()
    agent = SarsaSolver(env)
    rewards = agent.train(episodes=5)

    assert len(rewards) == 5
    assert all(isinstance(r, (float, int)) for r in rewards)


def test_policy_gradient_runs():
    env = AppalachianTrailEnv()
    agent = PolicyGradientSolver(env)
    rewards = agent.train(episodes=5)

    assert len(rewards) == 5
    assert all(isinstance(r, (float, int)) for r in rewards)


if __name__ == "__main__":
    print("Running solver tests...")
    test_q_learning_runs()
    test_sarsa_runs()
    test_policy_gradient_runs()
    print("All solver tests passed!")