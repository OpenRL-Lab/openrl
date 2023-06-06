import torch
import torch.nn as nn


class VDNBase(nn.Module):
    def __init__(self):
        super(VDNBase, self).__init__()

    def forward(self, agent_qs):
        return torch.sum(agent_qs, dim=1, keepdim=True)
