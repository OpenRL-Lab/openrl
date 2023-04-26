import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import get_clones, init


class Encoder(nn.Module):
    def __init__(self, cfg, split_shape, cat_self=True):
        super(Encoder, self).__init__()
        self._use_orthogonal = cfg.use_orthogonal
        self._activation_id = cfg.activation_id
        self._attn_N = cfg.attn_N
        self._attn_size = cfg.attn_size
        self._attn_heads = cfg.attn_heads
        self._dropout = cfg.dropout
        self._use_average_pool = cfg.use_average_pool
        self._cat_self = cat_self
        if self._cat_self:
            self.embedding = CatSelfEmbedding(
                split_shape[1:],
                self._attn_size,
                self._use_orthogonal,
                self._activation_id,
            )
        else:
            self.embedding = Embedding(
                split_shape[1:],
                self._attn_size,
                self._use_orthogonal,
                self._activation_id,
            )

        self.layers = get_clones(
            EncoderLayer(
                self._attn_size,
                self._attn_heads,
                self._dropout,
                self._use_orthogonal,
                self._activation_id,
            ),
            self._attn_N,
        )
        self.norm = nn.LayerNorm(self._attn_size)

    def forward(self, x, self_idx=-1, mask=None):
        x, self_x = self.embedding(x, self_idx)
        for i in range(self._attn_N):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        if self._use_average_pool:
            x = torch.transpose(x, 1, 2)
            x = F.avg_pool1d(x, kernel_size=x.size(-1)).view(x.size(0), -1)
            if self._cat_self:
                x = torch.cat((x, self_x), dim=-1)
        x = x.view(x.size(0), -1)
        return x


# [L,[1,2],[1,2],[1,2]]
def split_obs(obs, split_shape):
    start_idx = 0
    split_obs = []
    for i in range(len(split_shape)):
        split_obs.append(
            obs[:, start_idx : (start_idx + split_shape[i][0] * split_shape[i][1])]
        )
        start_idx += split_shape[i][0] * split_shape[i][1]
    return split_obs


class FeedForward(nn.Module):
    def __init__(
        self, d_model, d_ff=512, dropout=0.0, use_orthogonal=True, activation_id=1
    ):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id]
        )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.linear_1 = nn.Sequential(
            init_(nn.Linear(d_model, d_ff)), active_func, nn.LayerNorm(d_ff)
        )

        self.dropout = nn.Dropout(dropout)
        self.linear_2 = init_(nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = self.dropout(self.linear_1(x))
        x = self.linear_2(x)
        return x


def ScaledDotProductAttention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.0, use_orthogonal=True):
        super(MultiHeadAttention, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = init_(nn.Linear(d_model, d_model))
        self.v_linear = init_(nn.Linear(d_model, d_model))
        self.k_linear = init_(nn.Linear(d_model, d_model))
        self.dropout = nn.Dropout(dropout)
        self.out = init_(nn.Linear(d_model, d_model))

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention
        scores = ScaledDotProductAttention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        heads,
        dropout=0.0,
        use_orthogonal=True,
        activation_id=False,
        d_ff=512,
        use_FF=False,
    ):
        super(EncoderLayer, self).__init__()
        self._use_FF = use_FF
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout, use_orthogonal)
        self.ff = FeedForward(d_model, d_ff, dropout, use_orthogonal, activation_id)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        if self._use_FF:
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.ff(x2))
        return x


class CatSelfEmbedding(nn.Module):
    def __init__(self, split_shape, d_model, use_orthogonal=True, activation_id=1):
        super(CatSelfEmbedding, self).__init__()
        self.split_shape = split_shape

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id]
        )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        for i in range(len(split_shape)):
            if i == (len(split_shape) - 1):
                setattr(
                    self,
                    "fc_" + str(i),
                    nn.Sequential(
                        init_(nn.Linear(split_shape[i][1], d_model)),
                        active_func,
                        nn.LayerNorm(d_model),
                    ),
                )
            else:
                setattr(
                    self,
                    "fc_" + str(i),
                    nn.Sequential(
                        init_(
                            nn.Linear(split_shape[i][1] + split_shape[-1][1], d_model)
                        ),
                        active_func,
                        nn.LayerNorm(d_model),
                    ),
                )

    def forward(self, x, self_idx=-1):
        x = split_obs(x, self.split_shape)
        N = len(x)

        x1 = []
        self_x = x[self_idx]
        for i in range(N - 1):
            K = self.split_shape[i][0]
            L = self.split_shape[i][1]
            for j in range(K):
                torch.cat((x[i][:, (L * j) : (L * j + L)], self_x), dim=-1)
                exec("x1.append(self.fc_{}(temp))".format(i))
        x[self_idx]
        exec("x1.append(self.fc_{}(temp))".format(N - 1))

        out = torch.stack(x1, 1)

        return out, self_x


class Embedding(nn.Module):
    def __init__(self, split_shape, d_model, use_orthogonal=True, activation_id=1):
        super(Embedding, self).__init__()
        self.split_shape = split_shape

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(
            ["tanh", "relu", "leaky_relu", "leaky_relu"][activation_id]
        )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        for i in range(len(split_shape)):
            setattr(
                self,
                "fc_" + str(i),
                nn.Sequential(
                    init_(nn.Linear(split_shape[i][1], d_model)),
                    active_func,
                    nn.LayerNorm(d_model),
                ),
            )

    def forward(self, x, self_idx=None):
        x = split_obs(x, self.split_shape)
        N = len(x)

        x1 = []
        for i in range(N):
            K = self.split_shape[i][0]
            L = self.split_shape[i][1]
            for j in range(K):
                x[i][:, (L * j) : (L * j + L)]
                exec("x1.append(self.fc_{}(temp))".format(i))

        out = torch.stack(x1, 1)

        if self_idx is None:
            return out, None
        else:
            return out, x[self_idx]
