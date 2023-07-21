## Installation

### 1. Simulator
 
 Download [Nvidia Isaac Sim](https://developer.nvidia.com/isaac-sim) and install it.

Note 1: If you download the container version of Nvidia Isaac Sim running in the cloud, simulation interface can't be visualized. 

Note 2: Latest version Isaac Sim 2022.2.1 provides a built-in Python 3.7 environment that packages can use, similar to a system-level Python install. We recommend using this Python environment when running the Python scripts.

### 2. RL tasks
Install [Omniverse Isaac Gym Reinforcement Learning Environments for Isaac Sim](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs). This  repository contains Reinforcement Learning tasks that can be run with the latest release of Isaac Sim.

Please make sure you follow the above [repo](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs) and successfully run the training script:

``` shell
PYTHON_PATH scripts/rlgames_train.py task=Ant headless=True
```

## Usage

`cfg` folder provides Cartpole task configs in Isaac Sim following [Omniverse Isaac Gym Reinforcement Learning Environments for Isaac Sim](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs).

You can use `OpenRL` to train Isaac Sim Cartpole via:

``` shell
PYTHON_PATH train_ppo.py
```