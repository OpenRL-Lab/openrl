import torch
import torch.nn as nn

from .attention import Encoder
from .util import get_clones, init


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, activation_id):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id]
        )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
        )

        if self._layer_N > 1:
            self.fc_h = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)),
                active_func,
                nn.LayerNorm(hidden_size),
            )
            self.fc2 = get_clones(self.fc_h, self._layer_N - 1)

        self.fc3 = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N - 1):
            x = self.fc2[i](x)
        x = self.fc3(x)
        return x


class CONVLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, use_orthogonal, activation_id):
        super(CONVLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id]
        )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.conv = nn.Sequential(
            init_(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=hidden_size // 4,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                )
            ),
            active_func,  # nn.BatchNorm1d(hidden_size//4),
            init_(
                nn.Conv1d(
                    in_channels=hidden_size // 4,
                    out_channels=hidden_size // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ),
            active_func,  # nn.BatchNorm1d(hidden_size//2),
            init_(
                nn.Conv1d(
                    in_channels=hidden_size // 2,
                    out_channels=hidden_size,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ),
            active_func,
        )  # , nn.BatchNorm1d(hidden_size))

    def forward(self, x):
        x = self.conv(x)
        return x


class MLPBase(nn.Module):
    def __init__(self, cfg, obs_shape, use_attn_internal=False, use_cat_self=True):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = cfg.use_feature_normalization
        self._use_orthogonal = cfg.use_orthogonal
        self._activation_id = cfg.activation_id
        self._use_attn = cfg.use_attn
        self._use_attn_internal = use_attn_internal
        self._use_average_pool = cfg.use_average_pool
        self._use_conv1d = cfg.use_conv1d
        self._stacked_frames = cfg.stacked_frames
        self._layer_N = 0 if cfg.use_single_network else cfg.layer_N
        self._attn_size = cfg.attn_size
        self.hidden_size = cfg.hidden_size

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        if self._use_attn and self._use_attn_internal:
            if self._use_average_pool:
                if use_cat_self:
                    inputs_dim = self._attn_size + obs_shape[-1][1]
                else:
                    inputs_dim = self._attn_size
            else:
                split_inputs_dim = 0
                split_shape = obs_shape[1:]
                for i in range(len(split_shape)):
                    split_inputs_dim += split_shape[i][0]
                inputs_dim = split_inputs_dim * self._attn_size
            self.attn = Encoder(cfg, obs_shape, use_cat_self)
            self.attn_norm = nn.LayerNorm(inputs_dim)
        else:
            inputs_dim = obs_dim

        if self._use_conv1d:
            self.conv = CONVLayer(
                self._stacked_frames,
                self.hidden_size,
                self._use_orthogonal,
                self._activation_id,
            )
            random_x = torch.FloatTensor(
                1, self._stacked_frames, inputs_dim // self._stacked_frames
            )
            random_out = self.conv(random_x)
            assert len(random_out.shape) == 3
            inputs_dim = random_out.size(-1) * random_out.size(-2)

        self.mlp = MLPLayer(
            inputs_dim,
            self.hidden_size,
            self._layer_N,
            self._use_orthogonal,
            self._activation_id,
        )

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        if self._use_attn and self._use_attn_internal:
            x = self.attn(x, self_idx=-1)
            x = self.attn_norm(x)

        if self._use_conv1d:
            batch_size = x.size(0)
            x = x.view(batch_size, self._stacked_frames, -1)
            x = self.conv(x)
            x = x.view(batch_size, -1)

        x = self.mlp(x)

        return x

    @property
    def output_size(self):
        return self.hidden_size
