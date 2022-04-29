import torch
from ..models.CSWM import CSWM
from ..modules.TransitionGNN import TransitionGNN
from ..modules.AttentionV1 import AttentionV1
from ..constants import Constants
from ..utils import to_one_hot


class CSWMHardAttention(CSWM):

    def build_transition_(self, config):

        self.use_attention = True
        attention = config[Constants.ATTENTION]

        self.attention = AttentionV1(
            state_size=self.embedding_dim,
            action_size=self.action_dim,
            key_query_size=512,
            value_size=attention[Constants.VALUE_SIZE],
            sqrt_scale=True,
            use_sigmoid=False
        )

        self.transition_model = TransitionGNN(
            input_dim=self.embedding_dim,
            hidden_dim=self.gnn_hidden_dim,
            action_dim=self.action_dim,
            num_objects=self.num_objects,
            ignore_action=self.ignore_action,
            copy_action=self.copy_action,
            num_layers=self.num_gnn_layers
        )

    def energy(self, state, action, next_state, no_trans=False):
        """Energy function based on normalized squared L2 norm."""

        norm = 0.5 / (self.sigma**2)

        if no_trans:
            diff = state - next_state
            return norm * diff.pow(2).sum(2)
        else:
            pred_state, weights = self.forward_transition(state, action, all=True)
            diff = pred_state - next_state[:, None]
            diff = diff.pow(2).sum(3).mean(2)
            diff = (diff * weights).sum(1)
            return norm * diff

    def forward_transition(self, state, action, all=False):

        if len(action.shape) == 1:
            action = to_one_hot(action, self.action_dim)
        else:
            assert len(action.shape) == 2
        weights = self.attention.forward_weights([state, action])

        if all:
            pred_state = []
            for obj_idx in range(self.num_objects):
                node_idx = torch.zeros(action.size(0), dtype=torch.long, device=action.device) + obj_idx
                tmp_action = self.action_to_target_node(action, node_idx)
                pred_trans, _, _ = self.transition_model([state, tmp_action, False])
                pred_state.append(state + pred_trans)
            return torch.stack(pred_state, dim=1), weights
        else:
            node_idx = torch.argmax(weights, dim=1)
            action = self.action_to_target_node(action, node_idx)
            pred_trans, _, _ = self.transition_model([state, action, False])
            return state + pred_trans

    def forward_weights(self, state, action):

        if len(action.shape) == 1:
            action = to_one_hot(action, self.action_dim)
        else:
            assert len(action.shape) == 2
        return self.attention.forward_weights([state, action])

    def action_to_target_node(self, action, node_idx):

        new_action = torch.zeros(
            (action.size(0), self.num_objects, action.size(1)),
            dtype=torch.float32, device=action.device
        )
        indices = list(range(action.size(0)))
        new_action[indices, node_idx] = action.detach()
        return new_action
