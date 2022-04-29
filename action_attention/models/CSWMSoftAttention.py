from ..models.CSWM import CSWM
from ..modules.TransitionGNN import TransitionGNN
from ..modules.AttentionV1 import AttentionV1
from ..constants import Constants
from ..utils import to_one_hot


class CSWMSoftAttention(CSWM):

    def build_transition_(self, config):

        self.use_attention = True
        attention = config[Constants.ATTENTION]

        self.attention = AttentionV1(
            state_size=self.embedding_dim,
            action_size=self.action_dim,
            key_query_size=512,
            value_size=attention[Constants.VALUE_SIZE],
            sqrt_scale=True
        )
        tmp_action_dim = attention[Constants.VALUE_SIZE]

        self.transition_model = TransitionGNN(
            input_dim=self.embedding_dim,
            hidden_dim=self.gnn_hidden_dim,
            action_dim=tmp_action_dim,
            num_objects=self.num_objects,
            ignore_action=self.ignore_action,
            copy_action=self.copy_action,
            num_layers=self.num_gnn_layers
        )

    def forward_transition(self, state, action):

        if len(action.shape) == 1:
            action = to_one_hot(action, self.action_dim)
        else:
            assert len(action.shape) == 2
        action = self.attention([state, action])

        pred_trans, _, _ = self.transition_model([state, action, False])
        return state + pred_trans
