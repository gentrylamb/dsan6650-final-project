
# AppalachianRL

**Reinforcement Learning on the Appalachian Trail**

This project implements reinforcement learning environments and solvers for optimizing long-distance hiking decisions on the Appalachian Trail.
Agents must manage **energy**, **food**, **weather**, and **time** while choosing whether to hike, rest, or resupply.

The project includes:

* A custom **Gymnasium environment**
* Tabular RL solvers: **Q-Learning** and **SARSA**
* A deep RL **Policy Gradient** agent
* Unit tests for correctness and reproducibility

---

## Installation

Clone the repo and install in editable mode:

```bash
git clone <your-repo-url>
cd dsan6650-final-project
pip install -e .
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
│   └── trail.py              # AppalachianTrailEnv (Gymnasium environment)
│
├── solvers/
│   ├── base.py               # Base class for RL solvers
│   ├── q_learning.py         # Q-Learning implementation
│   ├── sarsa.py              # SARSA implementation
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
| `0`    | Hike 1 mile                              |
| `1`    | Hike 3 miles                             |
| `2`    | Hike 5 miles                             |
| `3`    | Rest (recover energy, consume food)      |
| `4`    | Resupply (only valid at resupply points) |

### Reward Signal

* `-1` per day (time pressure)
* `+0.5*miles` for hiking
* Heavy penalties for zero food/energy
* `+1000` for completing the trail

---

## Example: Running Q-Learning

```python
from AppalachianRL.envs.trail import AppalachianTrailEnv
from AppalachianRL.solvers.q_learning import QLearningSolver

env = AppalachianTrailEnv()
solver = QLearningSolver(env, alpha=0.1, gamma=0.99, epsilon=0.2)

solver.train(episodes=500)
solver.save("qtable.npy")
```

---

## Example: Running SARSA

```python
from AppalachianRL.envs.trail import AppalachianTrailEnv
from AppalachianRL.solvers.sarsa import SarsaSolver

env = AppalachianTrailEnv()
solver = SarsaSolver(env, alpha=0.1, gamma=0.99, epsilon=0.1)

solver.train(episodes=300)
```

---

## Example: Policy Gradient Agent

```python
from AppalachianRL.envs.trail import AppalachianTrailEnv
from AppalachianRL.solvers.policy_gradient import PolicyGradientSolver

env = AppalachianTrailEnv()
solver = PolicyGradientSolver(env, lr=1e-3, gamma=0.99)

solver.train(episodes=1000)
```

---

## Running Tests

From the repo root:

```bash
pytest -q
```

This runs:

* **Environment tests** (`test_env.py`)
* **Solver functional tests** (`test_solvers.py`)

---

## Future Work

* Add DQN, PPO, or A2C
* Add stochastic terrain (climbs, descents, fatigue)
* Add multi-day weather simulation
* Add gear weight, shelter planning, and town-stop decisions

---

## About

Created as a final project for Georgetown’s **DSAN-6650: Reinforcement Learning**, by **Gentry Lamb**.

---

If you want, I can also prepare:

* A project logo/banner
* Diagrams for the MDP (state, action, reward)
* A "How the environment works internally" section

Just tell me!
