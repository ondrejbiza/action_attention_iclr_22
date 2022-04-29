import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..utils import get_act_fn


class AttentionV1(nn.Module):

    HIDDEN_SIZE = 512

    def __init__(self, state_size, action_size, key_query_size, value_size, sqrt_scale,
                 ablate_weights=False, use_sigmoid=False):

        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.key_query_size = key_query_size
        self.value_size = value_size
        self.sqrt_scale = sqrt_scale
        self.ablate_weights = ablate_weights
        self.use_sigmoid = use_sigmoid

        if self.use_sigmoid:
            self.normalizer = lambda x, dim: F.sigmoid(x)
        else:
            self.normalizer = F.softmax

        self.fc_key = MLP(self.state_size, self.key_query_size, self.HIDDEN_SIZE)
        self.fc_query = MLP(self.action_size, self.key_query_size, self.HIDDEN_SIZE)
        self.fc_value = MLP(self.action_size, self.value_size, self.HIDDEN_SIZE)

    def forward(self, x):

        state, action = x

        # flatten state
        batch_size = state.size(0)
        obj_size = state.size(1)
        state_r = state.reshape(batch_size * obj_size, state.size(2))

        # create keys and queries
        key_r = self.fc_key(state_r)
        query = self.fc_query(action)
        value = self.fc_value(action)

        key = key_r.reshape(batch_size, obj_size, self.key_query_size)

        # compute a vector of attention weights, one for each object slot
        if self.sqrt_scale:
            weights = self.normalizer((key * query[:, None]).sum(dim=2) * (1 / np.sqrt(self.key_query_size)), dim=-1)
        else:
            weights = self.normalizer((key * query[:, None]).sum(dim=2), dim=-1)

        if self.ablate_weights:
            # set uniform weights to check if they provide any benefit
            weights = torch.ones_like(weights) / weights.shape[1]

        # create a separate action for each object slot
        # weights: [|B|, |O|], value: [|B|, value_size]
        # => [|B|, |O|, value_size]
        return weights[:, :, None] * value[:, None, :]

    def forward_weights(self, x):

        state, action = x

        # flatten state
        batch_size = state.size(0)
        obj_size = state.size(1)
        state_r = state.reshape(batch_size * obj_size, state.size(2))

        # create keys and queries
        key_r = self.fc_key(state_r)
        query = self.fc_query(action)

        key = key_r.reshape(batch_size, obj_size, self.key_query_size)

        # compute a vector of attention weights, one for each object slot
        if self.sqrt_scale:
            weights = self.normalizer((key * query[:, None]).sum(dim=2) * (1 / np.sqrt(self.key_query_size)), dim=-1)
        else:
            weights = self.normalizer((key * query[:, None]).sum(dim=2), dim=-1)

        if self.ablate_weights:
            # set uniform weights to check if they provide any benefit
            weights = torch.ones_like(weights) / weights.shape[1]

        return weights


class MLP(nn.Module):
    """MLP encoder, maps observation to latent state."""

    def __init__(self, input_dim, output_dim, hidden_dim, act_fn='relu'):
        super(MLP, self).__init__()

        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        self.act1 = get_act_fn(act_fn)
        self.act2 = get_act_fn(act_fn)

    def forward(self, x):
        h = self.act1(self.fc1(x))
        h = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)
