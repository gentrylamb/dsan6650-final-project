"""
AppalachianRL: Reinforcement Learning environments and solvers
for modeling thru-hiker decision-making on the Appalachian Trail.
"""

# Expose environments
from .envs.trail import AppalachianTrailEnv

# Expose tabular RL solvers
from .solvers.q_learning import QLearningSolver
from .solvers.sarsa import SarsaSolver

# Expose deep RL solver
from .solvers.policy_gradient import PolicyGradientSolver

__all__ = [
    "AppalachianTrailEnv",
    "QLearningSolver",
    "SarsaSolver",
    "PolicyGradientSolver",
]
