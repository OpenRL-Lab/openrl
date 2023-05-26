""""""
import numpy as np
from custom_registration import make

from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent


def train():
    # Create an environment. If multiple environments need to be run in parallel, set the asynchronous parameter to True.
    # If you need to specify a level, you can set the state parameter which is specific to each game.
    env = make("Airstriker-Genesis", state="Level1", env_num=2, asynchronous=True)
    # create the neural network
    net = Net(env, device="cuda")
    # initialize the trainer
    agent = Agent(net)
    # start training
    agent.train(total_time_steps=2000)
    # close the environment
    env.close()
    return agent


def game_test(agent):
    # begin to test
    env = make(
        "Airstriker-Genesis",
        state="Level1",
        render_mode="group_human",
        env_num=4,
        asynchronous=True,
    )
    agent.set_env(env)
    obs, info = env.reset()
    done = False
    step = 0
    while True:
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        step += 1
        print(f"{step}: reward:{np.mean(r)}")

        if any(done):
            break

    env.close()


if __name__ == "__main__":
    agent = train()
    game_test(agent)
