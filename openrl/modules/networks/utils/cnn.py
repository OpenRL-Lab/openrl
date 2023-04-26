import torch.nn as nn

from .util import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNLayer(nn.Module):
    def __init__(
        self,
        obs_shape,
        hidden_size,
        use_orthogonal,
        activation_id,
        kernel_size=3,
        stride=1,
    ):
        super(CNNLayer, self).__init__()

        [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id]
        )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]
        obs_shape[1]
        obs_shape[2]

        # self.cnn = nn.Sequential(
        #     init_(nn.Conv2d(in_channels=input_channel, out_channels=hidden_size//2, kernel_size=kernel_size, stride=stride)), active_func,
        #     Flatten(),
        #     init_(nn.Linear(hidden_size//2 * (input_width-kernel_size+stride) * (input_height-kernel_size+stride), hidden_size)), active_func,
        #     init_(nn.Linear(hidden_size, hidden_size)), active_func)
        # only for atari
        self.cnn = nn.Sequential(
            init_(nn.Conv2d(input_channel, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)

        return x


class CNNBase(nn.Module):
    def __init__(self, cfg, obs_shape):
        super(CNNBase, self).__init__()

        self._use_orthogonal = cfg.use_orthogonal
        self._activation_id = cfg.activation_id
        self.hidden_size = cfg.hidden_size

        self.cnn = CNNLayer(
            obs_shape, self.hidden_size, self._use_orthogonal, self._activation_id
        )

    def forward(self, x):
        x = self.cnn(x)
        return x

    @property
    def output_size(self):
        return self.hidden_size
