# -*- coding:utf-8  -*-
def sample_single_dim(action_space_list_each, is_act_continuous):
    if is_act_continuous:
        each = action_space_list_each.sample()
    else:
        if action_space_list_each.__class__.__name__ == "Discrete":
            each = [0] * action_space_list_each.n
            idx = action_space_list_each.sample()
            each[idx] = 1
        elif action_space_list_each.__class__.__name__ == "MultiDiscreteParticle":
            each = []
            nvec = action_space_list_each.high - action_space_list_each.low + 1
            sample_indexes = action_space_list_each.sample()

            for i in range(len(nvec)):
                dim = nvec[i]
                new_action = [0] * dim
                index = sample_indexes[i]
                new_action[index] = 1
                each.extend(new_action)
    return each


def my_controller(observation, action_space, is_act_continuous):
    joint_action = []
    for i in range(len(action_space)):
        player = sample_single_dim(action_space[i], is_act_continuous)
        joint_action.append(player)

    return joint_action
