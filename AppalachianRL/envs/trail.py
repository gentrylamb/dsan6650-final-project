import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AppalachianTrailEnv(gym.Env):
    """
    A simplified reinforcement learning environment simulating a thru-hike 
    of the Appalachian Trail. The agent must manage energy, food, and time 
    while deciding how far to hike, when to rest, and when to resupply.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 trail_length=2200,      # total distance (2,197.4 miles)
                 max_energy=100,
                 max_food=10,
                 resupply_points=None,
                 seed=None):

        super().__init__()
        self.trail_length = trail_length
        self.max_energy = max_energy
        self.max_food = max_food
        self.rng = np.random.default_rng(seed)

        # Default resupply every ~250 miles
        self.resupply_points = resupply_points or [i for i in range(0, trail_length, 250)]

        # --- Observation Space ---
        # [miles_remaining, energy, food, weather, day]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([trail_length, max_energy, max_food, 2, 365], dtype=np.float32),
            dtype=np.float32
        )

        # --- Action Space ---
        # 0 = hike 1 mile
        # 1 = hike 3 miles
        # 2 = hike 5 miles
        # 3 = rest
        # 4 = resupply (only valid at stops)
        self.action_space = spaces.Discrete(5)

        self.reset()

    # --------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.miles_remaining = float(self.trail_length)
        self.energy = float(self.max_energy)
        self.food = float(self.max_food)
        self.day = 0

        # 0 = clear, 1 = rain, 2 = storm
        self.weather = 0  

        return self._get_obs(), {}

    # --------------------------------------------------------

    def _get_obs(self):
        return np.array([
            self.miles_remaining,
            self.energy,
            self.food,
            self.weather,
            self.day
        ], dtype=np.float32)

    # --------------------------------------------------------

    def step(self, action):
        done = False
        info = {}
        reward = 0

        # Daily weather update
        self.weather = self.rng.choice([0, 1, 2], p=[0.7, 0.2, 0.1])  # mostly clear

        # ----------------------------------------------------
        # Action Effects
        # ----------------------------------------------------

        if action == 0:   # Hike 1 mile
            miles = 1
            energy_cost = 10
            food_cost = 0.2

        elif action == 1: # Hike 3 miles
            miles = 3
            energy_cost = 20
            food_cost = 0.5

        elif action == 2: # Hike 5 miles
            miles = 5
            energy_cost = 35
            food_cost = 0.8

        elif action == 3: # Rest
            miles = 0
            energy_cost = -20  # regain energy
            food_cost = 1      # full day of food consumption

        elif action == 4: # Resupply
            miles = 0
            energy_cost = -10  # resting while resupplying
            food_cost = 0
            if int(self.trail_length - self.miles_remaining) not in self.resupply_points:
                # invalid action: no resupply point here
                reward -= 10

            else:
                self.food = self.max_food  # full refill

        # Weather modifies energy cost
        if self.weather == 1:   # rain
            energy_cost *= 1.2
        elif self.weather == 2: # storm
            energy_cost *= 1.5

        # Apply transitions
        self.miles_remaining = max(0, self.miles_remaining - miles)
        self.energy = np.clip(self.energy - energy_cost, 0, self.max_energy)
        self.food = max(0, self.food - food_cost)
        self.day += 1

        # ----------------------------------------------------
        # Reward Structure
        # ----------------------------------------------------
        # Goal: minimize days, avoid running out of food/energy

        reward -= 1  # time penalty

        if miles > 0:
            reward += miles * 0.5  # small positive for progress

        if self.energy == 0:
            reward -= 50  # collapse risk

        if self.food == 0:
            reward -= 50  # starvation risk

        # Completion
        if self.miles_remaining <= 0:
            reward += 1000
            done = True

        # Failure
        if self.energy == 0 or self.food == 0 or self.day >= 365:
            done = True

        return self._get_obs(), reward, done, False, info

    # --------------------------------------------------------

    def render(self):
        print(f"Day {self.day}: {self.miles_remaining:.1f} miles left | "
              f"Energy {self.energy:.1f} | Food {self.food:.1f} | Weather {self.weather}")
