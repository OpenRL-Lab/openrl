## Gallery

In order to facilitate users' familiarity with the framework, we provide more examples and demos of using OpenRL in Gallery. 

Users are also welcome to contribute their own training examples and demos to the Gallery.

### Tags:

![MARL](https://img.shields.io/badge/-MARL-yellow)
![sparse](https://img.shields.io/badge/-sparse%20reward-orange)
![offline](https://img.shields.io/badge/-offlineRL-darkblue)
![selfplay](https://img.shields.io/badge/-selfplay-blue)
![mbrl](https://img.shields.io/badge/-ModelBasedRL-lightblue)

![discrete](https://img.shields.io/badge/-discrete-brightgreen) (Discrete Action Space)

![continuous](https://img.shields.io/badge/-continous-green) (Continuous Action Space)

![hybrid](https://img.shields.io/badge/-hybrid-darkgreen) (Discrete+Continuous Action Space)

![IL](https://img.shields.io/badge/-IL/SL-purple) (Imitation Learning or Supervised Learning）

## Algorithm List

<div align="center">

|                     Algorithm                     |                                                          Tags                                                           |              Refs               |
|:-------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------:|:-------------------------------:|
|      [PPO](https://arxiv.org/abs/1707.06347)      |                             ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                             |  [code](./examples/cartpole/)   |
|     [PPO-continuous](https://arxiv.org/abs/1707.06347)      |                             ![continuous](https://img.shields.io/badge/-continous-green)                             |  [code](./examples/mujoco/)    |
| [Dual-clip PPO](https://arxiv.org/abs/1912.09729) |                             ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                             |  [code](./examples/cartpole/)   |
|     [MAPPO](https://arxiv.org/abs/2103.01955)     |                             ![MARL](https://img.shields.io/badge/-MARL-yellow)                             |  [code](./examples/mpe/)   |
|     [JRPO](https://arxiv.org/abs/2302.07515)      |                             ![MARL](https://img.shields.io/badge/-MARL-yellow)                             |  [code](./examples/mpe/)   |
|      [MAT](https://arxiv.org/abs/2205.14953)      |                             ![MARL](https://img.shields.io/badge/-MARL-yellow)                             |  [code](./examples/mpe/)   |
</div>

## Demo List

<div align="center">

|                                                                                                   Environment/Demo                                                                                                    |                                                          Tags                                                           |              Refs               |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------:|:-------------------------------:|
|                                                  [MuJoCo](https://github.com/deepmind/mujoco)<br>  <img width="300px" height="auto" src="./docs/images/mujoco.png">                                                   |                             ![continuous](https://img.shields.io/badge/-continous-green)                             |   [code](./examples/mujoco/)    |
|                               [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)<br>  <img width="300px" height="auto" src="./docs/images/cartpole.png">                                |                             ![discrete](https://img.shields.io/badge/-discrete-brightgreen)                             |  [code](./examples/cartpole/)   |
|                       [MPE: Simple Spread](https://pettingzoo.farama.org/environments/mpe/simple_spread/)<br>  <img width="300px" height="auto" src="./docs/images/simple_spread_trained.gif">                        | ![discrete](https://img.shields.io/badge/-discrete-brightgreen)  ![MARL](https://img.shields.io/badge/-MARL-yellow) |     [code](./examples/mpe/)     |
| [Super Mario Bros](https://github.com/Kautenja/gym-super-mario-bros)<br>  <img width="300px" height="auto" src="https://user-images.githubusercontent.com/2184469/40948820-3d15e5c2-6830-11e8-81d4-ecfaffee0a14.png"> | ![discrete](https://img.shields.io/badge/-discrete-brightgreen)  | [code](./examples/super_mario/) |
|                                                [Gym Retro](https://github.com/openai/retro)<br>  <img width="300px" height="auto" src="./docs/images/gym-retro.webp">                                                 | ![discrete](https://img.shields.io/badge/-discrete-brightgreen)  |    [code](./examples/retro/)    |
</div>