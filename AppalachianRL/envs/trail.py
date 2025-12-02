# envs/trail.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class AppalachianTrailEnv(gym.Env):
    """
    A simplified reinforcement learning environment simulating a thru-hike 
    of the Appalachian Trail. The agent must manage energy, food, and time 
    while deciding how far to hike, when to rest, and when to resupply.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 trail_length=2200,    # Total distance (2,197.4 miles)
                 max_energy=100,       # Energy (percent of 100)
                 max_food=10,          # Number of days of food
                 max_days = 213,       # Approx. 7 months
                 seed=None):

        super().__init__()
        self.trail_length = trail_length
        self.max_energy = max_energy
        self.max_food = max_food
        self.max_days = max_days         
        self.rng = np.random.default_rng(seed)

        # Resupply points (https://whiteblaze.net/forum/content.php/1344-Resuppling-within-one-miles-from-the-Appalachian-Trail-for-a-thru-hike)
        self.resupply_points = [round(item) for item in pd.read_csv("data/resupply_points.csv")['mile'].tolist()]

        # --- Observation Space ---
        # [miles_remaining, energy, food, weather, day]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([trail_length, max_energy, max_food, 2, max_days], dtype=np.float32),
            dtype=np.float32
        )

        # --- Action Space ---
        # 0 = hike easy day (8-12 miles)
        # 1 = hike standard day (13-18 miles)
        # 2 = hike big day (19-25 miles)
        # 3 = rest/zero day (0 miles)
        # 4 = resupply day (only valid at stops)
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

    def _calculate_distance_to_resupply(self):
        # get current mile and find distance to closest resupply
        current_mile = self.trail_length - self.miles_remaining
        next_resupply = min([p for p in self.resupply_points if p > current_mile], default=float('inf'))

        return next_resupply - current_mile
    
    # --------------------------------------------------------

    def step(self, action):
        done = False
        info = {}
        reward = 0

        # Daily weather update
        self.weather = self.rng.choice([0, 1, 2], p=[0.7, 0.2, 0.1])  # mostly clear

        # dist to next resupply
        resupply_dist = self._calculate_distance_to_resupply()

        # ----------------------------------------------------
        # Action Effects
        # ----------------------------------------------------

        # All Hiking Actions
        if action in [0,1,2]:
            # Hike easy day (8-12 miles)
            if action == 0:
                miles = np.random.randint(8,13)
                energy_cost = 0.5*miles
                food_cost = 0.8 

            # Hike standard day (13-18 miles)
            elif action == 1: 
                miles = np.random.randint(13,19)
                energy_cost = 0.75*miles
                food_cost = 1

            # Hike big day (19-25 miles)
            elif action == 2: 
                miles = np.random.randint(18,26)
                energy_cost = 1*miles
                food_cost = 1.1

            # Make sure agent doesnt pass resupply stop
            if resupply_dist <= miles:
                print('   STOPPED: You stopped at a resupply point')
                miles = resupply_dist # stop at the point
                info["stopped_at_resuppply"] = True

        # Rest/Zero Day Action (0 miles)
        elif action == 3: 
            print(f"   REST: You rested at {self.trail_length - self.miles_remaining} miles.")
            miles = 0
            energy_cost = -80   # 80% recovery
            food_cost = 0.5

         # Resupply Day Action
        elif action == 4:
            if resupply_dist <= 0.5:
                # valid action
                print(f"   RESUPPLY: You resupplied at {self.trail_length - self.miles_remaining} miles.")
                miles = resupply_dist
                energy_cost = -50
                food_cost = 0
                # refill and reward
                self.food = self.max_food  # full refill
                reward += 150
            else:
                # invalid action
                print(f"   INVALID ACTION: No resupply point nearby ({resupply_dist} miles away).")
                miles = 0
                energy_cost = 0
                food_cost = 0
                # penalty
                reward -= 100

        # Weather modifies energy cost (only for expenditure)
        if energy_cost > 0:
            if self.weather == 1: energy_cost *= 1.1  # rain
            elif self.weather == 2: energy_cost *= 1.3 # storm
        
        # scale down for testing
        food_cost *= 0
        energy_cost *= 0

        # Apply transitions
        self.miles_remaining = max(0, self.miles_remaining - miles)
        self.energy = np.clip(self.energy - energy_cost, 0, self.max_energy)
        self.food = max(0, self.food - food_cost)
        self.day += 1

        # ----------------------------------------------------
        # Reward Structure
        # ----------------------------------------------------
        # Goal: minimize days, avoid running out of food/energy

        # time penalty
        reward -= 1 
        # progress reward
        if miles > 0: reward += miles * 1
        # penalties for low reserves
        if self.energy <= 5: reward -= 50  # collapse risk
        if self.food <= .5: reward -= 50  # starvation risk

        # Completion Case
        if self.miles_remaining <= 0:
            print("   SUCCESS: Congrats, you completed the AT!")
            info["failure_reason"] = "Success"
            reward += 1000
            done = True

        # Failure Cases
        if self.energy <= 0:
            print("   FAILURE: You ran out of energy!")
            info["failure_reason"] = "Energy"
            reward -= 100
            done = True

        if self.food <= 0:
            print("   FAILURE: You ran out of food!")
            info["failure_reason"] = "Food"
            reward -= 300
            done = True
            
        if self.day >= self.max_days:
            print("   FAILURE: You took too long!")
            info["failure_reason"] = "Time"
            reward -= 100
            done = True
        
        return self._get_obs(), reward, done, False, info

    # --------------------------------------------------------

    def render(self):
        print(f"Day {self.day}: {self.miles_remaining:.1f} miles left | "
              f"Energy {self.energy:.1f} | Food {self.food:.1f} | Weather {self.weather}")
