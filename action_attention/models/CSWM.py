# taken from https://github.com/tkipf/c-swm, authors T Kipf, E van der Pol
import numpy as np
import torch
from torch import nn
from ..modules.EncoderCNNSmall import EncoderCNNSmall
from ..modules.EncoderCNNMedium import EncoderCNNMedium
from ..modules.EncoderCNNLarge import EncoderCNNLarge
from ..modules.EncoderMLP import EncoderMLP
from ..modules.TransitionGNN import TransitionGNN
from ..constants import Constants


class CSWM(nn.Module):

    def __init__(self, config, logger):

        super().__init__()

        self.logger = logger
        self.pos_loss = 0
        self.neg_loss = 0
        self.read_config_(config)
        self.build_encoder_(config)
        self.build_transition_(config)

    def read_config_(self, config):

        c = config
        self.cnn_hidden_dim = c[Constants.CNN_HIDDEN_DIM]
        self.mlp_hidden_dim = c[Constants.MLP_HIDDEN_DIM]
        self.gnn_hidden_dim = c[Constants.GNN_HIDDEN_DIM]
        self.embedding_dim = c[Constants.EMBEDDING_DIM]
        self.action_dim = c[Constants.ACTION_DIM]
        self.num_objects = c[Constants.NUM_OBJECTS]
        self.hinge = c[Constants.HINGE]
        self.sigma = c[Constants.SIGMA]
        self.ignore_action = c[Constants.IGNORE_ACTION]
        self.copy_action = c[Constants.COPY_ACTION]
        self.num_gnn_layers = c[Constants.NUM_GNN_LAYERS]

    def build_encoder_(self, config):

        encoder = config[Constants.ENCODER]
        num_channels = 3

        if encoder == 'small':
            self.obj_extractor = EncoderCNNSmall(
                input_dim=num_channels,
                hidden_dim=self.cnn_hidden_dim,
                num_objects=self.num_objects)
            # CNN image size changes
            width_height = np.array([50, 50])
            width_height = width_height // 10
        elif encoder == 'medium':
            num_channels = 6
            self.obj_extractor = EncoderCNNMedium(
                input_dim=num_channels,
                hidden_dim=self.cnn_hidden_dim,
                num_objects=self.num_objects)
            # CNN image size changes
            width_height = np.array([50, 50])
            width_height = width_height // 5
        elif encoder == 'large':
            self.obj_extractor = EncoderCNNLarge(
                input_dim=num_channels,
                hidden_dim=self.cnn_hidden_dim,
                num_objects=self.num_objects)
            width_height = np.array([50, 50])

        self.obj_encoder = EncoderMLP(
            input_dim=np.prod(width_height),
            hidden_dim=self.mlp_hidden_dim,
            output_dim=self.embedding_dim,
            num_objects=self.num_objects)

    def build_transition_(self, config):

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
            pred_state = self.forward_transition(state, action)
            diff = pred_state - next_state
            return norm * diff.pow(2).sum(2).mean(1)

    def transition_loss(self, state, action, next_state):

        return self.energy(state, action, next_state).mean()

    def contrastive_loss(self, obs, action, next_obs):

        state, next_state = self.extract_objects_(obs, next_obs)

        self.pos_loss = self.energy(state, action, next_state)
        self.pos_loss = self.pos_loss.mean()

        # Sample negative state across episodes at random
        neg_obs, neg_state = self.create_negatives_(obs, state)

        self.negative_loss_(state, neg_state)
        self.neg_loss = self.neg_loss.mean()

        loss = self.pos_loss + self.neg_loss

        return loss

    def extract_objects_(self, obs, next_obs):

        state = self.forward(obs)
        next_state = self.forward(next_obs)

        return state, next_state

    def create_negatives_(self, obs, state):

        batch_size = state.size(0)
        perm = np.random.permutation(batch_size)
        neg_obs = obs[perm]
        neg_state = state[perm]

        return neg_obs, neg_state

    def negative_loss_(self, state, neg_state):

        # [B, |O|]
        energy = self.energy(state, None, neg_state, no_trans=True)

        self.neg_loss = self.hinge - energy
        zeros = torch.zeros_like(self.neg_loss)
        self.neg_loss = torch.max(zeros, self.neg_loss)

    def forward(self, obs):

        return self.obj_encoder(self.obj_extractor(obs))

    def forward_transition(self, state, action):

        pred_trans, _, _ = self.transition_model([state, action, False])
        return state + pred_trans
