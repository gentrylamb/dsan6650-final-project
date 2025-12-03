# AppalachianRL

**Reinforcement Learning on the Appalachian Trail**

This project implements reinforcement learning environments and solvers for optimizing long-distance hiking decisions on the Appalachian Trail.
Agents must manage **energy**, **food**, **weather**, and **time** while choosing whether to hike, rest, or resupply.

The project includes:

* A custom **Gymnasium environment**
* Tabular RL solvers: **Q-Learning** and **SARSA**
* Pollicy-based solvers: Deep RL **Policy Gradient** agent and **Soft Actor-Critic (SAC)**
* Unit tests for correctness and reproducibility
* A notebook file to showcase the training results of the solvers 

---

## Installation

Clone the repo and install in editable mode:

```bash
git clone <your-repo-url>
cd dsan6650-final-project
pip install -e ./AppalachianRL/
```

Dependencies (auto-installed):

* Python 3.10+
* gymnasium
* numpy
* torch

---

## Project Structure

```
AppalachianRL/
│
├── envs/
│   ├── trail.py              # AppalachianTrailEnv (Gymnasium environment)
│   └── trail_adv.py          # Advanced implementation
│
├── solvers/
│   ├── base.py               # Base class for RL solvers
│   ├── random.py             # Random actions as baseline
│   ├── q_learning.py         # Q-Learning implementation
│   ├── sarsa.py              # SARSA implementation
│   ├── actor_critic.py       # Actor-Critic implementation
│   └── policy_gradient.py    # Deep RL policy gradient solver
│
├── tests/
│   ├── test_env.py           # Unit tests for the environment
│   └── test_solvers.py       # Unit tests for Q-learning & SARSA
│
├── __init__.py
└── pyproject.toml
```

---

## The Environment

`AppalachianTrailEnv` simulates a simplified thru-hike:

| Attribute         | Description               |
| ----------------- | ------------------------- |
| `miles_remaining` | Distance left to Katahdin |
| `energy`          | Physical energy (0-100)   |
| `food`            | Food stores (0-10 days)   |
| `weather`         | 0=clear, 1=rain, 2=storm  |
| `day`             | Days since start          |

### Actions

| Action | Meaning                                  |
| ------ | ---------------------------------------- |
| `0`    | Hike small day (8-12 miles)              |
| `1`    | Hike small day (13-18 miles)             |
| `2`    | Hike small day (19-25+ miles)            |
| `3`    | Rest / Zero Day                          |
| `4`    | Resupply Day (only valid at points)      |

### Reward Signal

* Small negative per day (time pressure)
* Small positive for progress
* Heavy penalties for zero food/energy
* large positive for completing the trail

---

## Example: Running Q-Learning

```python
from AppalachianRL.envs.trail import AppalachianTrailEnv
from AppalachianRL.solvers.q_learning import QLearningSolver

env = AppalachianTrailEnv()
solver = QLearningSolver(env, alpha=0.1, gamma=0.99, epsilon=0.2)

solver.train(episodes=500)
```

---

## Running Tests

From the repo root:

```bash
# test that env works as expected
python AppalachianRL/tests/test_env.py

# test that solvers function as expected
python AppalachianRL/tests/test_solvers.py
```

---

## Future Work

* Add DQN, PPO, or A2C
* Add stochastic terrain (climbs, descents, fatigue)
* Add multi-day weather simulation
* Add gear weight, shelter planning, and town-stop decisions

---

## About

Created as a final project for Georgetown’s **DSAN-6650: Reinforcement Learning**, by **Gentry Lamb**.
