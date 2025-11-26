# solvers/base_solver.py
from abc import ABC, abstractmethod

class BaseSolver(ABC):
    def __init__(self, env):
        self.env = env
    
    @abstractmethod
    def train(self, episodes: int):
        pass
    
    @abstractmethod
    def act(self, state):
        pass
