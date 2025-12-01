# envs/trail.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class AppalachianTrailEnv2(gym.Env):
    """
    Appalachian Trail environment WITHOUT energy.
    Agent manages:
      - distance
      - food
      - weather
      - time
    Resupply happens automatically when reaching town mile markers.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 trail_length=2200,
                 max_food=10,       # 10 days is realistic max capacity
                 max_days=213,      # 7 months
                 seed=None):

        super().__init__()
        self.trail_length = trail_length
        self.max_food = max_food
        self.max_days = max_days
        self.rng = np.random.default_rng(seed)

        # AT town mile markers
        self.resupply_points = sorted(
            round(m) for m in pd.read_csv("data/resupply_points.csv")["mile"].tolist()
        )

        # ----- Observation Space -----
        # miles_remaining, food_left, weather, day
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([trail_length, max_food, 2, max_days], dtype=np.float32),
            dtype=np.float32
        )

        # ----- Action Space -----
        # Choose hike intensity or rest
        self.action_space = spaces.Discrete(4)
        # 0 = easy (7–10 miles)
        # 1 = normal (11–15 miles)
        # 2 = push (16–22 miles)
        # 3 = zero day (0 miles)

        self.reset()

    # -------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.miles_remaining = float(self.trail_length)
        self.food = float(self.max_food)
        self.day = 0
        self.weather = 0  # 0 clear, 1 rain, 2 storm

        return self._get_obs(), {}

    # -------------------------------------------------------------

    def _get_obs(self):
        return np.array([
            self.miles_remaining,
            self.food,
            self.weather,
            self.day
        ], dtype=np.float32)

    # -------------------------------------------------------------

    def step(self, action):
        done = False
        reward = 0
        info = {}

        # Weather for the day
        self.weather = self.rng.choice([0, 1, 2], p=[0.7, 0.2, 0.1])

        # Pre-step tracking (for resupply)
        last_mile = self.trail_length - self.miles_remaining

        # -------------------------
        # Hiking / resting dynamics
        # -------------------------
        if action == 0:     # easy day
            miles = self.rng.integers(7, 11)
            food_cost = 0.7

        elif action == 1:   # normal day
            miles = self.rng.integers(11, 16)
            food_cost = 1.0

        elif action == 2:   # push day
            miles = self.rng.integers(16, 23)
            food_cost = 1.4

        elif action == 3:   # zero day
            miles = 0
            food_cost = 0.5

        # Weather modifies distance, not energy (since removed)
        # if self.weather == 1:     # rain
        #     miles = int(miles * 0.9)
        # elif self.weather == 2:   # storm
        #     miles = int(miles * 0.75)

        # Apply transitions
        self.miles_remaining = max(0, self.miles_remaining - miles)
        self.food = max(0, self.food - food_cost)
        self.day += 1

        # -------------------------
        # Automatic resupply
        # -------------------------
        current_mile = self.trail_length - self.miles_remaining
        for p in self.resupply_points:
            if last_mile < p <= current_mile:
                self.food = self.max_food
                reward += 8       # small positive signal
                info["resupply"] = True
                break

        # -------------------------
        # Reward shaping
        # -------------------------
        reward += miles * 0.75       # progress reward
        reward -= 1                  # time penalty

        if self.food <= 1:
            reward -= 10             # low food warning

        # -------------------------
        # Termination conditions
        # -------------------------
        if self.miles_remaining <= 0:
            reward += 400
            done = True

        if self.food <= 0:
            reward -= 100
            done = True

        if self.day >= self.max_days:
            reward -= 100
            done = True

        return self._get_obs(), reward, done, False, info

    # -------------------------------------------------------------

    def render(self):
        print(
            f"Day {self.day}: {self.miles_remaining:.1f} miles left | "
            f"Food {self.food:.1f} | Weather {self.weather}"
        )