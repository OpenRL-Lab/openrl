import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from openrl.buffers.utils.util import get_critic_obs_space, get_policy_obs_space
from openrl.modules.networks.base_value_policy_network import BaseValuePolicyNetwork
from openrl.modules.networks.utils.transformer_act import (
    continuous_autoregreesive_act,
    continuous_parallel_act,
    discrete_autoregreesive_act,
    discrete_parallel_act,
)
from openrl.modules.networks.utils.util import init
from openrl.utils.util import check_v2 as check


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain("relu")
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(n_agent + 1, n_agent + 1)).view(
                1, 1, n_agent + 1, n_agent + 1
            ),
        )

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        )  # (B, nh, L, hs)
        q = (
            self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        )  # (B, nh, L, hs)
        v = (
            self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        )  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, L, D)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # self.attn = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd)),
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd)),
        )

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class Encoder(nn.Module):
    def __init__(
        self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state
    ):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state
        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            init_(nn.Linear(state_dim, n_embd), activate=True),
            nn.GELU(),
        )
        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(obs_dim),
            init_(nn.Linear(obs_dim, n_embd), activate=True),
            nn.GELU(),
        )

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(
            *[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)]
        )
        self.head = nn.Sequential(
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, 1)),
        )

    def forward(self, state, obs):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        if self.encode_state:
            state_embeddings = self.state_encoder(state)
            x = state_embeddings
        else:
            obs_embeddings = self.obs_encoder(obs)
            x = obs_embeddings

        rep = self.blocks(self.ln(x))
        v_loc = self.head(rep)

        return v_loc, rep


class Decoder(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        n_block,
        n_embd,
        n_head,
        n_agent,
        action_type="Discrete",
        dec_actor=False,
        share_actor=False,
    ):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type

        if action_type != "Discrete":
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

        if self.dec_actor:
            if self.share_actor:
                print("mac_dec!!!!!")
                self.mlp = nn.Sequential(
                    nn.LayerNorm(obs_dim),
                    init_(nn.Linear(obs_dim, n_embd), activate=True),
                    nn.GELU(),
                    nn.LayerNorm(n_embd),
                    init_(nn.Linear(n_embd, n_embd), activate=True),
                    nn.GELU(),
                    nn.LayerNorm(n_embd),
                    init_(nn.Linear(n_embd, action_dim)),
                )
            else:
                self.mlp = nn.ModuleList()
                for n in range(n_agent):
                    actor = nn.Sequential(
                        nn.LayerNorm(obs_dim),
                        init_(nn.Linear(obs_dim, n_embd), activate=True),
                        nn.GELU(),
                        nn.LayerNorm(n_embd),
                        init_(nn.Linear(n_embd, n_embd), activate=True),
                        nn.GELU(),
                        nn.LayerNorm(n_embd),
                        init_(nn.Linear(n_embd, action_dim)),
                    )
                    self.mlp.append(actor)
        else:
            # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))
            if action_type == "Discrete":
                self.action_encoder = nn.Sequential(
                    init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                    nn.GELU(),
                )
            else:
                self.action_encoder = nn.Sequential(
                    init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU()
                )
            self.obs_encoder = nn.Sequential(
                nn.LayerNorm(obs_dim),
                init_(nn.Linear(obs_dim, n_embd), activate=True),
                nn.GELU(),
            )
            self.ln = nn.LayerNorm(n_embd)
            self.blocks = nn.Sequential(
                *[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)]
            )
            self.head = nn.Sequential(
                init_(nn.Linear(n_embd, n_embd), activate=True),
                nn.GELU(),
                nn.LayerNorm(n_embd),
                init_(nn.Linear(n_embd, action_dim)),
            )

    def zero_std(self, device):
        if self.action_type != "Discrete":
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action, obs_rep, obs):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(obs)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](obs[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else:
            action_embeddings = self.action_encoder(action)
            x = self.ln(action_embeddings)
            for block in self.blocks:
                x = block(x, obs_rep)
            logit = self.head(x)

        return logit


class MultiAgentTransformer(BaseValuePolicyNetwork):
    def __init__(
        self,
        cfg,
        input_space,
        action_space,
        device=torch.device("cpu"),
        use_half=False,
        extra_args=None,
    ):
        assert not use_half, "half precision not supported for MAT algorithm"
        super(MultiAgentTransformer, self).__init__(cfg, device)

        obs_dim = get_policy_obs_space(input_space)[0]
        critic_obs_dim = get_critic_obs_space(input_space)[0]

        n_agent = cfg.num_agents
        n_block = cfg.n_block
        n_embd = cfg.n_embd
        n_head = cfg.n_head
        encode_state = cfg.encode_state
        dec_actor = cfg.dec_actor
        share_actor = cfg.share_actor
        if action_space.__class__.__name__ == "Box":
            self.action_type = "Continuous"
            action_dim = action_space.shape[0]
            self.action_num = action_dim
        else:
            self.action_type = "Discrete"
            action_dim = action_space.n
            self.action_num = 1

        self.n_agent = n_agent
        self.obs_dim = obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self._use_policy_active_masks = cfg.use_policy_active_masks
        self.device = device

        # state unused
        state_dim = 37

        self.encoder = Encoder(
            state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state
        )
        self.decoder = Decoder(
            obs_dim,
            action_dim,
            n_block,
            n_embd,
            n_head,
            n_agent,
            self.action_type,
            dec_actor=dec_actor,
            share_actor=share_actor,
        )
        self.to(device)

    def zero_std(self):
        if self.action_type != "Discrete":
            self.decoder.zero_std(self.device)

    def eval_actions(
        self, obs, rnn_states, action, masks, action_masks=None, active_masks=None
    ):
        obs = obs.reshape(-1, self.n_agent, self.obs_dim)

        action = action.reshape(-1, self.n_agent, self.action_num)

        if action_masks is not None:
            action_masks = action_masks.reshape(-1, self.n_agent, self.action_dim)

        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # action_masks: (batch, n_agent, act_dim)

        # state unused
        ori_shape = np.shape(obs)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if action_masks is not None:
            action_masks = check(action_masks).to(**self.tpdv)

        batch_size = np.shape(state)[0]
        v_loc, obs_rep = self.encoder(state, obs)
        if self.action_type == "Discrete":
            action = action.long()
            action_log, entropy = discrete_parallel_act(
                self.decoder,
                obs_rep,
                obs,
                action,
                batch_size,
                self.n_agent,
                self.action_dim,
                self.tpdv,
                action_masks,
            )
        else:
            action_log, entropy = continuous_parallel_act(
                self.decoder,
                obs_rep,
                obs,
                action,
                batch_size,
                self.n_agent,
                self.action_dim,
                self.tpdv,
            )
        action_log = action_log.view(-1, self.action_num)
        v_loc = v_loc.view(-1, 1)
        entropy = entropy.view(-1, self.action_num)
        if self._use_policy_active_masks and active_masks is not None:
            entropy = (entropy * active_masks).sum() / active_masks.sum()
        else:
            entropy = entropy.mean()
        return action_log, entropy, v_loc

    def get_actions(
        self,
        obs,
        rnn_states_actor=None,
        masks=None,
        action_masks=None,
        deterministic=False,
    ):
        obs = obs.reshape(-1, self.n_agent, self.obs_dim)
        if action_masks is not None:
            action_masks = action_masks.reshape(-1, self.num_agents, self.action_dim)

        # state unused
        ori_shape = np.shape(obs)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if action_masks is not None:
            action_masks = check(action_masks).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        v_loc, obs_rep = self.encoder(state, obs)
        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_autoregreesive_act(
                self.decoder,
                obs_rep,
                obs,
                batch_size,
                self.n_agent,
                self.action_dim,
                self.tpdv,
                action_masks,
                deterministic,
            )
        else:
            output_action, output_action_log = continuous_autoregreesive_act(
                self.decoder,
                obs_rep,
                obs,
                batch_size,
                self.n_agent,
                self.action_dim,
                self.tpdv,
                deterministic,
            )

        output_action = output_action.reshape(-1, output_action.shape[-1])
        output_action_log = output_action_log.reshape(-1, output_action_log.shape[-1])
        return output_action, output_action_log, None

    def get_values(self, critic_obs, rnn_states_critic=None, masks=None):
        critic_obs = critic_obs.reshape(-1, self.n_agent, self.critic_obs_dim)

        ori_shape = np.shape(critic_obs)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(critic_obs).to(**self.tpdv)
        v_tot, obs_rep = self.encoder(state, obs)

        v_tot = v_tot.reshape(-1, v_tot.shape[-1])
        return v_tot, None

    def get_actor_para(self):
        return self.parameters()

    def get_critic_para(self):
        return self.parameters()
