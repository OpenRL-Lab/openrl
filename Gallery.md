## Gallery

In order to facilitate users' familiarity with the framework, we provide more examples and demos of using OpenRL in Gallery. 

Users are also welcome to contribute their own training examples and demos to the Gallery.

### Tags:

![MARL](https://img.shields.io/badge/-MARL-yellow)
![NLP](https://img.shields.io/badge/-NLP-green)
![Transformer](https://img.shields.io/badge/-Transformer-blue)
![sparse](https://img.shields.io/badge/-sparse%20reward-orange)
![offline](https://img.shields.io/badge/-offlineRL-darkblue)
![selfplay](https://img.shields.io/badge/-selfplay-blue)
![mbrl](https://img.shields.io/badge/-ModelBasedRL-lightblue)
![image](https://img.shields.io/badge/-image-red)

![value](https://img.shields.io/badge/-value-orange) (Value-based RL)

![offpolicy](https://img.shields.io/badge/-offpolicy-blue) (Off-policy RL)


![discrete](https://img.shields.io/badge/-discrete-brightgreen) (Discrete Action Space)

![continuous](https://img.shields.io/badge/-continous-green) (Continuous Action Space)

![hybrid](https://img.shields.io/badge/-hybrid-darkgreen) (Discrete+Continuous Action Space)

![IL](https://img.shields.io/badge/-IL/SL-purple) (Imitation Learning or Supervised Learning）

## Algorithm List

<div align="center">

|                               Algorithm                                |                                                                                        Tags                                                                                         |               Refs                |
|:----------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------:|
|                [PPO](https://arxiv.org/abs/1707.06347)                 |                                                           ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                                           |   [code](./examples/cartpole/)    |
|           [PPO-continuous](https://arxiv.org/abs/1707.06347)           |                                                            ![continuous](https://img.shields.io/badge/-continous-green)                                                             |    [code](./examples/mujoco/)     |
|           [Dual-clip PPO](https://arxiv.org/abs/1912.09729)            |                                                           ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                                                           |   [code](./examples/cartpole/)    |
|               [MAPPO](https://arxiv.org/abs/2103.01955)                |                                                                 ![MARL](https://img.shields.io/badge/-MARL-yellow)                                                                  |      [code](./examples/mpe/)      |
|                [JRPO](https://arxiv.org/abs/2302.07515)                |                                                                 ![MARL](https://img.shields.io/badge/-MARL-yellow)                                                                  |      [code](./examples/mpe/)      |
|                [GAIL](https://arxiv.org/abs/1606.03476)                |                                ![offline](https://img.shields.io/badge/-offlineRL-darkblue)                                                                                         |     [code](./examples/gail/)      |
| [Behavior Cloning](http://www.cse.unsw.edu.au/~claude/papers/MI15.pdf) |                                ![offline](https://img.shields.io/badge/-offlineRL-darkblue)                                                                                         | [code](./examples/behavior_cloning/) |
|                 [DQN](https://arxiv.org/abs/1312.5602)                 | ![discrete](https://img.shields.io/badge/-discrete-brightgreen)   ![value](https://img.shields.io/badge/-value-orange)   ![offpolicy](https://img.shields.io/badge/-offpolicy-blue) |   [code](./examples/gridworld/)   |
|                [MAT](https://arxiv.org/abs/2205.14953)                 |                                 ![MARL](https://img.shields.io/badge/-MARL-yellow)  ![Transformer](https://img.shields.io/badge/-Transformer-blue)                                  |      [code](./examples/mpe/)      |
|                [VDN](https://arxiv.org/abs/1706.05296)                 |                                 ![MARL](https://img.shields.io/badge/-MARL-yellow)                                  |      [code](./examples/mpe/)      |
|                [DDPG](https://arxiv.org/abs/1509.02971)                |                                 ![continuous](https://img.shields.io/badge/-continous-green)                                  |     [code](./examples/ddpg/)      |
|                               Self-Play                                |                              ![selfplay](https://img.shields.io/badge/-selfplay-blue) ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                               |   [code](./examples/selfplay/)    |
</div>

## Demo List

<div align="center">

|                                                                                                   Environment/Demo                                                                                                    |                                                        Tags                                                         |              Refs               |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------:|:-------------------------------:|
|                                                  [MuJoCo](https://github.com/deepmind/mujoco)<br>  <img width="300px" height="auto" src="./docs/images/mujoco.png">                                                   |                            ![continuous](https://img.shields.io/badge/-continous-green)                             |   [code](./examples/mujoco/)    |
|                               [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)<br>  <img width="300px" height="auto" src="./docs/images/cartpole.png">                                |                           ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                           |  [code](./examples/cartpole/)   |
|                       [MPE: Simple Spread](https://pettingzoo.farama.org/environments/mpe/simple_spread/)<br>  <img width="300px" height="auto" src="./docs/images/simple_spread_trained.gif">                        | ![discrete](https://img.shields.io/badge/-discrete-brightgreen)  ![MARL](https://img.shields.io/badge/-MARL-yellow) |     [code](./examples/mpe/)     |
|                                                  [StarCraft II](https://github.com/oxwhirl/smac)<br>  <img width="300px" height="auto" src="./docs/images/smac.png">                                                  | ![discrete](https://img.shields.io/badge/-discrete-brightgreen)  ![MARL](https://img.shields.io/badge/-MARL-yellow) |    [code](./examples/smac/)     |
|                                [Chat Bot](https://openrl-docs.readthedocs.io/en/latest/quick_start/train_nlp.html)<br>  <img width="300px" height="auto" src="./docs/images/chat.gif">                                |                          ![discrete](https://img.shields.io/badge/-discrete-brightgreen)        ![NLP](https://img.shields.io/badge/-NLP-green)     ![Transformer](https://img.shields.io/badge/-Transformer-blue)                               |     [code](./examples/nlp/)     |
|                                        [Atari Pong](https://gymnasium.farama.org/environments/atari/pong/)<br>  <img width="300px" height="auto" src="./docs/images/pong.png">                                        |                          ![discrete](https://img.shields.io/badge/-discrete-brightgreen)        ![image](https://img.shields.io/badge/-image-red)                                    |    [code](./examples/atari/)    |
|                                   [PettingZoo: Tic-Tac-Toe](https://pettingzoo.farama.org/environments/classic/tictactoe/)<br>  <img width="300px" height="auto" src="./docs/images/tic-tac-toe.jpeg">                                    |                      ![selfplay](https://img.shields.io/badge/-selfplay-blue)    ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                               |  [code](./examples/selfplay/)   |
|                                   [Omniverse Isaac Gym](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs)<br>  <img width="300px" height="auto" src="https://user-images.githubusercontent.com/34286328/171454189-6afafbff-bb61-4aac-b518-24646007cb9f.gif">                                    |                       ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                               |    [code](./examples/isaac/)    |
|                                                      [GridWorld](./examples/gridworld/)<br>  <img width="300px" height="auto" src="./docs/images/gridworld.jpg">                                                      |                          ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                               |  [code](./examples/gridworld/)  |
| [Super Mario Bros](https://github.com/Kautenja/gym-super-mario-bros)<br>  <img width="300px" height="auto" src="https://user-images.githubusercontent.com/2184469/40948820-3d15e5c2-6830-11e8-81d4-ecfaffee0a14.png"> |                           ![discrete](https://img.shields.io/badge/-discrete-brightgreen)     ![image](https://img.shields.io/badge/-image-red)                      | [code](./examples/super_mario/) |
|                                                 [Gym Retro](https://github.com/openai/retro)<br>  <img width="300px" height="auto" src="./docs/images/gym-retro.jpg">                                                 |                           ![discrete](https://img.shields.io/badge/-discrete-brightgreen)     ![image](https://img.shields.io/badge/-image-red)                      |    [code](./examples/retro/)    |
</div>