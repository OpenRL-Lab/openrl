<div align="center">
    <a href="https://openrl-docs.readthedocs.io/zh/latest/index.html"><img width="450px" height="auto" src="docs/images/openrl_text.png"></a>
</div>

---
[![PyPI](https://img.shields.io/pypi/v/openrl)](https://pypi.org/project/openrl/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/openrl)
[![Anaconda-Server Badge](https://anaconda.org/openrl/openrl/badges/version.svg)](https://anaconda.org/openrl/openrl)
[![Anaconda-Server Badge](https://anaconda.org/openrl/openrl/badges/latest_release_date.svg)](https://anaconda.org/openrl/openrl)
[![Anaconda-Server Badge](https://anaconda.org/openrl/openrl/badges/downloads.svg)](https://anaconda.org/openrl/openrl)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


[![Hits-of-Code](https://hitsofcode.com/github/OpenRL-Lab/openrl?branch=main)](https://hitsofcode.com/github/OpenRL-Lab/openrl/view?branch=main)
[![codecov](https://codecov.io/gh/OpenRL-Lab/openrl_release/branch/main/graph/badge.svg?token=4FMEYMR83U)](https://codecov.io/gh/OpenRL-Lab/openrl_release)

[![Documentation Status](https://readthedocs.org/projects/openrl-docs/badge/?version=latest)](https://openrl-docs.readthedocs.io/en/latest/?badge=latest)
[![Read the Docs](https://img.shields.io/readthedocs/openrl-docs-zh?label=%E4%B8%AD%E6%96%87%E6%96%87%E6%A1%A3)](https://openrl-docs.readthedocs.io/zh/latest/)

![GitHub Org's stars](https://img.shields.io/github/stars/OpenRL-Lab)
[![GitHub stars](https://img.shields.io/github/stars/OpenRL-Lab/openrl)](https://github.com/opendilab/OpenRL/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/OpenRL-Lab/openrl)](https://github.com/OpenRL-Lab/openrl/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/OpenRL-Lab/openrl)
[![GitHub issues](https://img.shields.io/github/issues/OpenRL-Lab/openrl)](https://github.com/OpenRL-Lab/openrl/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/OpenRL-Lab/openrl)](https://github.com/OpenRL-Lab/openrl/pulls)
[![Contributors](https://img.shields.io/github/contributors/OpenRL-Lab/openrl)](https://github.com/OpenRL-Lab/openrl/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/OpenRL-Lab/openrl)](https://github.com/OpenRL-Lab/openrl/blob/master/LICENSE)

[![Embark](https://img.shields.io/badge/discord-OpenRL-%237289da.svg?logo=discord)](https://discord.gg/qfPBcVvT)
[![slack badge](https://img.shields.io/badge/Slack-join-blueviolet?logo=slack&amp)](https://join.slack.com/t/openrlhq/shared_invite/zt-1tqwpvthd-Eeh0IxQ~DIaGqYXoW2IUQg)

OpenRL-v0.0.15 is updated on July 21, 2023

The main branch is the latest version of OpenRL, which is under active development. If you just want to have a try with OpenRL, you can switch to the stable branch.

## Welcome to OpenRL

[Documentation](https://openrl-docs.readthedocs.io/en/latest/) | [中文介绍](README_zh.md) |  [中文文档](https://openrl-docs.readthedocs.io/zh/latest/)

<div align="center">
    Crafting Reinforcement Learning Frameworks with Passion, Your Valuable Insights Welcome.   <br><br>
</div>

OpenRL is an open-source general reinforcement learning research framework that supports training for various tasks 
such as single-agent, multi-agent, offline RL, and natural language. 
Developed based on PyTorch, the goal of OpenRL is to provide a simple-to-use, flexible, efficient and sustainable platform for the reinforcement learning research community.

Currently, the features supported by OpenRL include:

- A simple-to-use universal interface that supports training for both single-agent and multi-agent

- Support for offline RL training with expert dataset

- Reinforcement learning training support for natural language tasks (such as dialogue)

- Importing models and datasets from [Hugging Face](https://huggingface.co/)

- Support for models such as LSTM, GRU, Transformer etc.

- Multiple training acceleration methods including automatic mixed precision training and data collecting wth half precision policy network

- User-defined training models, reward models, training data and environment support

- Support for [gymnasium](https://gymnasium.farama.org/) environments

- Support for [Callbacks](https://openrl-docs.readthedocs.io/en/latest/callbacks/index.html), which can be used to implement various functions such as logging, saving, and early stopping

- Dictionary observation space support

- Popular visualization tools such as [wandb](https://wandb.ai/),  [tensorboardX](https://tensorboardx.readthedocs.io/en/latest/index.html) are supported

- Serial or parallel environment training while ensuring consistent results in both modes

- Chinese and English documentation

- Provides unit testing and code coverage testing

- Compliant with Black Code Style guidelines and type checking

Algorithms currently supported by OpenRL (for more details, please refer to [Gallery](./Gallery.md)):
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- [Dual-clip PPO](https://arxiv.org/abs/1912.09729)
- [Multi-agent PPO (MAPPO)](https://arxiv.org/abs/2103.01955)
- [Joint-ratio Policy Optimization (JRPO)](https://arxiv.org/abs/2302.07515)
- [Generative Adversarial Imitation Learning (GAIL)](https://arxiv.org/abs/1606.03476)
- [Behavior Cloning (BC)](http://www.cse.unsw.edu.au/~claude/papers/MI15.pdf)
- [Deep Q-Network (DQN)](https://arxiv.org/abs/1312.5602)
- [Multi-Agent Transformer (MAT)](https://arxiv.org/abs/2205.14953)
- [Value-Decomposition Network (VDN)](https://arxiv.org/abs/1706.05296)
- [Soft Actor Critic (SAC)](https://arxiv.org/abs/1812.05905)
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971)

Environments currently supported by OpenRL (for more details, please refer to [Gallery](./Gallery.md)):
- [Gymnasium](https://gymnasium.farama.org/)
- [MuJoCo](https://github.com/deepmind/mujoco)
- [MPE](https://github.com/openai/multiagent-particle-envs)
- [Chat Bot](https://openrl-docs.readthedocs.io/en/latest/quick_start/train_nlp.html)
- [Atari](https://gymnasium.farama.org/environments/atari/)
- [StarCraft II](https://github.com/oxwhirl/smac)
- [PettingZoo](https://pettingzoo.farama.org/)
- [Omniverse Isaac Gym](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs)
- [GridWorld](./examples/gridworld/)
- [Super Mario Bros](https://github.com/Kautenja/gym-super-mario-bros)
- [Gym Retro](https://github.com/openai/retro)

This framework has undergone multiple iterations by the [OpenRL-Lab](https://github.com/OpenRL-Lab) team which has applied it in academic research. 
It has now become a mature reinforcement learning framework.

OpenRL-Lab will continue to maintain and update OpenRL, and we welcome everyone to join our [open-source community](./CONTRIBUTING.md) 
to contribute towards the development of reinforcement learning.

For more information about OpenRL, please refer to the [documentation](https://openrl-docs.readthedocs.io/en/latest/).

## Outline

- [Welcome to OpenRL](#welcome-to-openrl)
- [Outline](#outline)
- [Installation](#installation)
- [Use Docker](#use-docker)
- [Quick Start](#quick-start)
- [Gallery](#gallery)
- [Projects Using OpenRL](#projects-using-openrl)
- [Feedback and Contribution](#feedback-and-contribution)
- [Maintainers](#maintainers)
- [Supporters](#supporters)
  - [&#8627; Contributors](#-contributors) 
  - [&#8627; Stargazers](#-stargazers)
  - [&#8627; Forkers](#-forkers)
- [Citing OpenRL](#citing-openrl)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

Users can directly install OpenRL via pip:

```bash
pip install openrl
```

If users are using Anaconda or Miniconda, they can also install OpenRL via conda:

```bash
conda install -c openrl openrl
```

Users who want to modify the source code can also install OpenRL from the source code:

```bash
git clone https://github.com/OpenRL-Lab/openrl.git && cd openrl
pip install -e .
```

After installation, users can check the version of OpenRL through command line:

```bash
openrl --version
```

**Tips**: No installation required, try OpenRL online through Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15VBA-B7AJF8dBazzRcWAxJxZI7Pl9m-g?usp=sharing)

## Use Docker

OpenRL currently provides Docker images with and without GPU support. 
If the user's computer does not have an NVIDIA GPU, they can obtain an image without the GPU plugin using the following command:
```bash
sudo docker pull openrllab/openrl-cpu
```

If the user wants to accelerate training with a GPU, they can obtain it using the following command:
```bash
sudo docker pull openrllab/openrl
```

After successfully pulling the image, users can run OpenRL's Docker image using the following commands:
```bash
# Without GPU acceleration
sudo docker run -it openrllab/openrl-cpu
# With GPU acceleration 
sudo docker run -it --gpus all --net host openrllab/openrl
```

Once inside the Docker container, users can check OpenRL's version and then run test cases using these commands: 
```bash 
# Check OpenRL version in Docker container  
openrl --version  
# Run test case  
openrl --mode train --env CartPole-v1  
```

## Quick Start

OpenRL provides a simple and easy-to-use interface for beginners in reinforcement learning. 
Below is an example of using the PPO algorithm to train the `CartPole` environment:
```python
# train_ppo.py
from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent
env = make("CartPole-v1", env_num=9) # Create an environment and set the environment parallelism to 9.
net = Net(env) # Create neural network.
agent = Agent(net) # Initialize the agent.
agent.train(total_time_steps=20000) # Start training and set the total number of steps to 20,000 for the running environment.
```
Training an agent using OpenRL only requires four simple steps: 
**Create Environment** => **Initialize Model** => **Initialize Agent** => **Start Training**!

For a well-trained agent, users can also easily test the agent:
```python
# train_ppo.py
from openrl.envs.common import make
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent
agent = Agent(Net(make("CartPole-v1", env_num=9))) # Initialize trainer.
agent.train(total_time_steps=20000)
# Create an environment for test, set the parallelism of the environment to 9, and set the rendering mode to group_human.
env = make("CartPole-v1", env_num=9, render_mode="group_human")
agent.set_env(env) # The agent requires an interactive environment.
obs, info = env.reset() # Initialize the environment to obtain initial observations and environmental information.
while True:
    action, _ = agent.act(obs) # The agent predicts the next action based on environmental observations.
    # The environment takes one step according to the action, obtains the next observation, reward, whether it ends and environmental information.
    obs, r, done, info = env.step(action)
    if any(done): break
env.close() # Close test environment
```
Executing the above code on a regular laptop only takes **a few seconds**
to complete the training. Below shows the visualization of the agent:

<div align="center">
  <img src="docs/images/train_ppo_cartpole.gif"></a>
</div>


**Tips:** Users can also quickly train the `CartPole` environment by executing a command line in the terminal.
```bash
openrl --mode train --env CartPole-v1
```

For training tasks such as multi-agent and natural language processing, OpenRL also provides a similarly simple and easy-to-use interface.

For information on how to perform multi-agent training, set hyperparameters for training, load training configurations, use wandb, save GIF animations, etc., please refer to:
- [Multi-Agent Training Example](https://openrl-docs.readthedocs.io/en/latest/quick_start/multi_agent_RL.html)

For information on natural language task training, loading models/datasets on Hugging Face, customizing training models/reward models, etc., please refer to:
- [Dialogue Task Training Example](https://openrl-docs.readthedocs.io/en/latest/quick_start/train_nlp.html)

For more information about OpenRL, please refer to the [documentation](https://openrl-docs.readthedocs.io/en/latest/).

## Gallery

In order to facilitate users' familiarity with the framework, we provide more examples and demos of using OpenRL in [Gallery](./Gallery.md). 
Users are also welcome to contribute their own training examples and demos to the Gallery.

## Projects Using OpenRL

We have listed research projects that use OpenRL in the [OpenRL Project](./Project.md). 
If you are using OpenRL in your research project, you are also welcome to join this list.

## Feedback and Contribution
- If you have any questions or find bugs, you can check or ask in the [Issues](https://github.com/OpenRL-Lab/openrl/issues).
- Join the QQ group: [OpenRL Official Communication Group](docs/images/qq.png)
<div align="center">
<a href="docs/images/qq.png"><img width="250px" height="auto" src="docs/images/qq.png"></a>
</div>

- Join the [slack](https://join.slack.com/t/openrlhq/shared_invite/zt-1tqwpvthd-Eeh0IxQ~DIaGqYXoW2IUQg) group to discuss OpenRL usage and development with us.
- Join the [Discord](https://discord.gg/qfPBcVvT) group to discuss OpenRL usage and development with us.
- Send an E-mail to: [huangshiyu@4paradigm.com](huangshiyu@4paradigm.com)
- Join the [GitHub Discussion](https://github.com/orgs/OpenRL-Lab/discussions).

The OpenRL framework is still under continuous development and documentation. 
We welcome you to join us in making this project better:
- How to contribute code: Read the [Contributors' Guide](./CONTRIBUTING.md)
- [OpenRL Roadmap](https://github.com/OpenRL-Lab/openrl/issues/2)

## Maintainers

At present, OpenRL is maintained by the following maintainers:
- [Shiyu Huang](https://huangshiyu13.github.io/)([@huangshiyu13](https://github.com/huangshiyu13))
- Wenze Chen([@Chen001117](https://github.com/Chen001117))
- Yiwen Sun([@YiwenAI](https://github.com/YiwenAI))

Welcome more contributors to join our maintenance team (send an E-mail to [huangshiyu@4paradigm.com](huangshiyu@4paradigm.com) 
to apply for joining the OpenRL team).

## Supporters

### &#8627; Contributors

<a href="https://github.com/OpenRL-Lab/openrl/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OpenRL-Lab/openrl" />
</a>

### &#8627; Stargazers

[![Stargazers repo roster for @OpenRL-Lab/openrl](https://reporoster.com/stars/OpenRL-Lab/openrl)](https://github.com/OpenRL-Lab/openrl/stargazers)

### &#8627; Forkers

[![Forkers repo roster for @OpenRL-Lab/openrl](https://reporoster.com/forks/OpenRL-Lab/openrl)](https://github.com/OpenRL-Lab/openrl/network/members)

## Citing OpenRL

If our work has been helpful to you, please feel free to cite us:
```latex
@misc{openrl2023,
    title={OpenRL},
    author={OpenRL Contributors},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/OpenRL-Lab/openrl}},
    year={2023},
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=OpenRL-Lab/openrl&type=Date)](https://star-history.com/#OpenRL-Lab/openrl&Date)

## License
OpenRL under the Apache 2.0 license.

## Acknowledgments
The development of the OpenRL framework has drawn on the strengths of other reinforcement learning frameworks:

- Stable-baselines3: https://github.com/DLR-RM/stable-baselines3
- pytorch-a2c-ppo-acktr-gail: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
- MAPPO: https://github.com/marlbenchmark/on-policy
- Gymnasium: https://github.com/Farama-Foundation/Gymnasium
- DI-engine: https://github.com/opendilab/DI-engine/
- Tianshou: https://github.com/thu-ml/tianshou
- RL4LMs: https://github.com/allenai/RL4LMs
