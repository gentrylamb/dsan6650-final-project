# envs/trail.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class AppalachianTrailEnv(gym.Env):
    """
    A simplified reinforcement learning environment simulating a thru-hike 
    of the Appalachian Trail. 
    
    UPDATED: Resupply is no longer a discrete action. It occurs automatically 
    when the agent hikes into a town (resupply point).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 trail_length=2200,    # Total distance (2,197.4 miles)
                 max_energy=100,       # Energy (percent of 100)
                 max_food=10,          # Number of days of food
                 max_days = 243,       # Approx. 8 months (generous buffer)
                 seed=None,
                 verbose=False):

        super().__init__()
        self.trail_length = trail_length
        self.max_energy = max_energy
        self.max_food = max_food
        self.max_days = max_days         
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

        # Resupply points
        self.resupply_points = sorted([round(item) for item in pd.read_csv("data/resupply_points.csv")['mile'].tolist()])

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
        # REMOVED: 4 = resupply day
        self.action_space = spaces.Discrete(4)

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
    
    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    # --------------------------------------------------------

    def step(self, action):
        done = False
        info = {}
        reward = 0

        # Daily weather update
        self.weather = self.rng.choice([0, 1, 2], p=[0.7, 0.2, 0.1])

        # Current State Calculations
        current_mile = self.trail_length - self.miles_remaining
        
        # Init vars
        miles = 0
        energy_cost = 0
        food_cost = 0
        hit_town = False

        # ----------------------------------------------------
        # Action Logic
        # ----------------------------------------------------

        # --- HIKING ACTIONS (0, 1, 2) ---
        if action in [0, 1, 2]:
            # Define distances
            if action == 0:   # Easy
                miles = np.random.randint(8, 13)
                energy_cost = 0.5 * miles
                food_cost = 0.8 
            elif action == 1: # Standard
                miles = np.random.randint(13, 19)
                energy_cost = 0.75 * miles
                food_cost = 1.0
            elif action == 2: # Big
                miles = np.random.randint(18, 26)
                energy_cost = 1.0 * miles
                food_cost = 1.2

            # --- AUTO-RESUPPLY CHECK ---
            # Check if this hike pushes us past (or onto) a resupply point
            projected_mile = current_mile + miles
            
            # Find closest point strictly ahead of us, but within range of the hike
            next_stops = [p for p in self.resupply_points if p > current_mile and p <= projected_mile]
            
            if next_stops:
                # We hit a town!
                target_stop = min(next_stops) # Stop at the first town encountered
                
                # Clamp miles so we stop exactly at the town
                miles = target_stop - current_mile
                
                hit_town = True
                self._vprint(f"   TOWN REACHED: Stopped at mile {target_stop}. Resupplying...")

        # --- REST ACTION (3) ---
        elif action == 3: 
            self._vprint(f"   REST: You rested at mile {current_mile}.")
            miles = 0
            energy_cost = -80   # Recover energy
            food_cost = 0.5

        # ----------------------------------------------------
        # Apply Costs & Modifiers
        # ----------------------------------------------------

        # Weather penalty on energy (only if moving)
        if miles > 0 and energy_cost > 0:
            if self.weather == 1: energy_cost *= 1.1  # rain
            elif self.weather == 2: energy_cost *= 1.3 # storm
        
        # Update State
        self.miles_remaining = max(0, self.miles_remaining - miles)
        self.day += 1
        
        # Apply food/energy costs
        self.energy = np.clip(self.energy - energy_cost, 0, self.max_energy)
        self.food = max(0, self.food - food_cost)

        # ----------------------------------------------------
        # Trigger Resupply Effects (If Town Reached)
        # ----------------------------------------------------
        if hit_town:
            # Add extra time
            # self.day += 0.1
            # Full Restore
            self.food = self.max_food
            self.energy = self.max_energy 
            reward += 50  # Bonus for making it to a safe haven
            info['resupplied'] = True

        # ----------------------------------------------------
        # Reward Structure
        # ----------------------------------------------------
        
        # 1. Progress Reward
        if miles > 0: 
            reward += miles * 0.5

        # 2. Time Penalty (Living Cost)
        reward -= 2 

        # 3. Danger Penalties (Do not die)
        if self.energy <= 10: reward -= 20
        if self.food <= 1: reward -= 20

        # --- TERMINATION CONDITIONS ---

        # 1. Success
        if self.miles_remaining <= 0:
            self._vprint(f"   SUCCESS: Completed in {self.day} days!")
            info["failure_reason"] = "Success"
            reward += 2000
            done = True

        # 2. Failure: Energy
        elif self.energy <= 0:
            self._vprint(f"   FAILURE: Collapsed from exhaustion at mile {current_mile}.")
            info["failure_reason"] = "Energy"
            reward -= 500
            done = True

        # 3. Failure: Food
        elif self.food <= 0:
            self._vprint(f"   FAILURE: Starved at mile {current_mile}.")
            info["failure_reason"] = "Food"
            reward -= 500
            done = True
            
        # 4. Failure: Time
        elif self.day >= self.max_days:
            self._vprint("   FAILURE: Winter came. You took too long.")
            info["failure_reason"] = "Time"
            reward -= 500
            done = True
        
        return self._get_obs(), reward, done, False, info

    # --------------------------------------------------------

    def render(self):
        print(f"Day {self.day}: {self.miles_remaining:.1f} miles left | "
              f"Energy {self.energy:.1f} | Food {self.food:.1f} | Weather {self.weather}")