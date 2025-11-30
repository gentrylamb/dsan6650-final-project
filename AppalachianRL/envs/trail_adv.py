# envs/trail_adv.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class AppalachianTrailAdvEnv(gym.Env):
    """
    A more challenging reinforcement learning environment simulating a thru-hike 
    of the Appalachian Trail. The agent must manage energy, food, fatigue, and weather 
    while deciding how far to hike, when to rest, forage, camp, or resupply.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 trail_length=500,     # Total distance (shortened for faster experiments)
                 max_energy=100,       # Energy percent
                 max_food=10,          # Number of days of food
                 max_days=183,         # Max episode length
                 seed=None):

        super().__init__()
        self.trail_length = trail_length
        self.max_energy = max_energy
        self.max_food = max_food
        self.max_days = max_days
        self.rng = np.random.default_rng(seed)

        # Load resupply points
        self.resupply_points = [round(item) for item in pd.read_csv("data/resupply_points.csv")['mile'].tolist()]

        # Observation: miles_remaining, energy, food, fatigue, weather, day
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([trail_length, max_energy, max_food, 100, 2, max_days], dtype=np.float32),
            dtype=np.float32
        )

        # Action space:
        # 0 = slow hike, 1 = normal hike, 2 = fast hike
        # 3 = rest, 4 = forage, 5 = camp, 6 = resupply
        self.action_space = spaces.Discrete(7)

        self.reset()

    # ------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.miles_remaining = float(self.trail_length)
        self.energy = float(self.max_energy)
        self.food = float(self.max_food)
        self.fatigue = 0
        self.day = 0
        self.weather = 0  # 0=clear,1=rain,2=storm
        return self._get_obs(), {}

    # ------------------------
    def _get_obs(self):
        return np.array([
            self.miles_remaining,
            self.energy,
            self.food,
            self.fatigue,
            self.weather,
            self.day
        ], dtype=np.float32)

    # ------------------------
    def _distance_to_resupply(self):
        current_mile = self.trail_length - self.miles_remaining
        next_stop = min([p for p in self.resupply_points if p > current_mile], default=None)
        return next_stop - current_mile if next_stop is not None else self.trail_length

    # ------------------------
    def step(self, action):
        done = False
        reward = 0
        info = {}

        # Weather update
        self.weather = self.rng.choice([0,1,2], p=[0.6,0.25,0.15])

        # -------------------
        # Action effects
        # -------------------
        miles = 0
        energy_cost = 0
        food_cost = 0
        fatigue_gain = 0

        if action == 0:   # slow hike
            miles = self.rng.integers(5,9)
            energy_cost = 0.8*miles
            fatigue_gain = 5
            food_cost = 0.5

        elif action == 1: # normal hike
            miles = self.rng.integers(10,15)
            energy_cost = 1.0*miles
            fatigue_gain = 10
            food_cost = 1.0

        elif action == 2: # fast hike
            miles = self.rng.integers(16,22)
            energy_cost = 1.3*miles
            fatigue_gain = 20
            food_cost = 1.2

        elif action == 3: # rest
            energy_cost = -50
            fatigue_gain = -20
            food_cost = 0.5

        elif action == 4: # forage
            food_gain = self.rng.integers(1,3)
            self.food = min(self.max_food, self.food + food_gain)
            energy_cost = -20
            fatigue_gain = -10

        elif action == 5: # camp (recover energy/fatigue)
            energy_cost = -70
            fatigue_gain = -50
            food_cost = 1

        elif action == 6: # resupply
            dist = self._distance_to_resupply()
            if dist > 5:
                reward -= 25  # invalid action
                miles = 0
            else:
                self.food = self.max_food
                energy_cost = -30
                fatigue_gain = -20
                miles = 0

        # Weather impact
        if self.weather == 1:  # rain
            energy_cost *= 1.2
            fatigue_gain *= 1.1
            reward -= 2
        elif self.weather == 2:  # storm
            energy_cost *= 1.5
            fatigue_gain *= 1.3
            reward -= 5

        # Apply transitions
        self.miles_remaining = max(0, self.miles_remaining - miles)
        self.energy = np.clip(self.energy - energy_cost, 0, self.max_energy)
        self.fatigue = np.clip(self.fatigue + fatigue_gain, 0, 100)
        self.food = max(0, self.food - food_cost)
        self.day += 1

        # -------------------
        # Reward shaping
        # -------------------
        reward += miles * 1.5                 # progress
        reward -= self.fatigue * 0.2          # fatigue penalty
        reward += self.energy*0.1              # energy bonus
        reward -= (self.max_food - self.food)*0.3 # low food penalty

        # Death/failure
        if self.energy <= 0:
            reward -= 50
            done = True
        if self.food <= 0:
            reward -= 50
            done = True
        if self.day >= self.max_days:
            done = True

        # Completion
        if self.miles_remaining <= 0:
            reward += 100
            done = True

        return self._get_obs(), reward, done, False, info

    # ------------------------
    def render(self):
        print(f"Day {self.day}: {self.miles_remaining:.1f} miles | "
              f"Energy {self.energy:.1f} | Food {self.food:.1f} | "
              f"Fatigue {self.fatigue:.1f} | Weather {self.weather}")