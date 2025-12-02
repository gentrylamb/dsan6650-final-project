"""
AppalachianRL: Reinforcement Learning environments and solvers
for modeling thru-hiker decision-making on the Appalachian Trail.
"""

# Expose environments
from .envs.trail import AppalachianTrailEnv

# Expose random solver
from .solvers.random import RandomAgent

# Expose tabular RL solvers
from .solvers.q_learning import QLearningSolver
from .solvers.sarsa import SarsaSolver

# Expose deep RL solver
from .solvers.policy_gradient import PolicyGradientSolver
from .solvers.actor_critic import ActorCriticSolver

__all__ = [
    "AppalachianTrailEnv",
    "QLearningSolver",
    "SarsaSolver",
    "PolicyGradientSolver",
    "ActorCriticSolver", 
    "RandomAgent"
]
