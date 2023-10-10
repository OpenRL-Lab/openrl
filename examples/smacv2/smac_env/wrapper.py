import random

import numpy as np

from .distributions import get_distribution
from .multiagentenv import MultiAgentEnv
from .StarCraft2_Env import StarCraft2Env


class StarCraftCapabilityEnvWrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.distribution_config = kwargs["capability_config"]
        self.env_key_to_distribution_map = {}
        self._parse_distribution_config()
        self.env = StarCraft2Env(**kwargs)
        assert (
            self.distribution_config.keys() == kwargs["capability_config"].keys()
        ), "Must give distribution config and capability config the same keys"

    def _parse_distribution_config(self):
        for env_key, config in self.distribution_config.items():
            if env_key == "n_units" or env_key == "n_enemies":
                continue
            config["env_key"] = env_key
            # add n_units key
            config["n_units"] = self.distribution_config["n_units"]
            config["n_enemies"] = self.distribution_config["n_enemies"]
            distribution = get_distribution(config["dist_type"])(config)
            self.env_key_to_distribution_map[env_key] = distribution

    def reset(self):
        reset_config = {}
        for distribution in self.env_key_to_distribution_map.values():
            reset_config = {**reset_config, **distribution.generate()}
        # print("成功reset一次")
        return self.env.reset(reset_config)

    def __getattr__(self, name):
        if hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            raise AttributeError

    def get_obs(self):
        return self.env.get_obs()

    def get_obs_feature_names(self):
        return self.env.get_obs_feature_names()

    def get_state(self):
        return self.env.get_state()

    def get_state_feature_names(self):
        return self.env.get_state_feature_names()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_env_info(self):
        return self.env.get_env_info()

    def get_obs_size(self):
        return self.env.get_obs_size()

    def get_state_size(self):
        return self.env.get_state_size()

    def get_total_actions(self):
        return self.env.get_total_actions()

    def get_capabilities(self):
        return self.env.get_capabilities()

    def get_obs_agent(self, agent_id):
        return self.env.get_obs_agent(agent_id)

    def get_avail_agent_actions(self, agent_id):
        return self.env.get_avail_agent_actions(agent_id)

    def render(self):
        return self.env.render()

    def step(self, actions):
        reward, terminated, info = self.env.step(actions)
        local_obs = self.get_obs()
        global_state = np.array(
            [
                self.env.get_state_agent(agent_id)
                for agent_id in range(self.env.n_agents)
            ]
        )
        rewards = [[reward]] * self.env.n_agents
        dones = []
        action_masks = []

        for i in range(self.env.n_agents):
            if terminated:
                dones.append(True)
            else:
                dones.append(self.env.death_tracker_ally[i])

        bad_transition = (
            True if self.env._episode_steps >= self.env.episode_limit else False
        )

        bad_transition_list = []
        battles_won_list = []
        battles_game_list = []
        battles_draw_list = []
        restarts_list = []
        won_list = []
        for i in range(self.env.n_agents):
            action_masks.append(self.get_avail_agent_actions(i))
            bad_transition_list.append(bad_transition)
            battles_won_list.append(self.env.battles_won)
            battles_game_list.append(self.env.battles_game)
            battles_draw_list.append(self.env.timeouts)
            restarts_list.append(self.env.force_restarts)
            won_list.append(self.env.win_counted)

        info["bad_transition"] = bad_transition_list
        info["battles_won"] = battles_won_list
        info["battles_game"] = battles_game_list
        info["battles_draw"] = battles_draw_list
        info["restarts"] = restarts_list
        info["won"] = won_list
        # print("成功step一次")
        return local_obs, global_state, rewards, dones, info, action_masks

    def get_stats(self):
        return self.env.get_stats()

    def full_restart(self):
        return self.env.full_restart()

    def save_replay(self):
        self.env.save_replay()

    def close(self):
        return self.env.close()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
