# taken from https://github.com/tkipf/c-swm, authors T Kipf, E van der Pol
import torch
from torch import nn
from ..utils import unsorted_segment_sum, to_one_hot, get_act_fn


class TransitionGNN(torch.nn.Module):
    """GNN-based transition function."""
    def __init__(self, input_dim, hidden_dim, action_dim, num_objects, ignore_action=False, copy_action=False,
                 act_fn='relu', layer_norm=True, num_layers=3):

        super(TransitionGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        self.ignore_action = ignore_action
        self.copy_action = copy_action
        self.num_layers = num_layers

        if self.ignore_action:
            self.action_dim = 0
        else:
            self.action_dim = action_dim

        tmp_action_dim = self.action_dim
        edge_mlp_input_size = self.input_dim * 2

        self.edge_mlp = nn.Sequential(*self.make_node_mlp_layers_(
            edge_mlp_input_size, self.hidden_dim, act_fn, layer_norm
        ))

        if self.num_objects == 1:
            node_input_dim = self.input_dim + tmp_action_dim
        else:
            node_input_dim = hidden_dim + self.input_dim + tmp_action_dim

        self.node_mlp = nn.Sequential(*self.make_node_mlp_layers_(
            node_input_dim, self.input_dim, act_fn, layer_norm
        ))

        self.edge_list = None
        self.batch_size = 0

    def _edge_model(self, source, target, action=None):

        if action is None:
            x = [source, target]
        else:
            x = [source, target, action]

        out = torch.cat(x, dim=1)

        return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, col = edge_index
            agg = unsorted_segment_sum(
                edge_attr, row, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr

        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_objects, device):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_objects, num_objects)

            # Remove diagonal.
            adj_full -= torch.eye(num_objects)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            offset = torch.arange(
                0, batch_size * num_objects, num_objects).unsqueeze(-1)
            offset = offset.expand(batch_size, num_objects * (num_objects - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)
            self.edge_list = self.edge_list.to(device)

        return self.edge_list

    def process_action_(self, action, viz=False):

        if self.copy_action:
            if len(action.shape) == 1:
                # action is an integer
                action_vec = to_one_hot(action, self.action_dim).repeat(1, self.num_objects)
            else:
                # action is a vector
                action_vec = action.repeat(1, self.num_objects)

            # mix node and batch dimension
            action_vec = action_vec.reshape(-1, self.action_dim).float()
        else:
            # we have a separate action for each node
            if len(action.shape) == 1:
                # index for both object and action
                action_vec = to_one_hot(action, self.action_dim * self.num_objects)
                action_vec = action_vec.reshape(-1, self.action_dim)
            else:
                action_vec = action.reshape(action.size(0), self.action_dim * self.num_objects)
                action_vec = action_vec.reshape(-1, self.action_dim)

        return action_vec

    def forward(self, x):

        states, action, viz = x

        device = states.device
        batch_size = states.size(0)
        num_nodes = states.size(1)

        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr = states.reshape(-1, self.input_dim)

        if not self.ignore_action:
            action_vec = self.process_action_(action, viz=viz)

        edge_attr = None
        edge_index = None

        if num_nodes > 1:
            # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(
                batch_size, num_nodes, device)

            row, col = edge_index
            edge_attr = self._edge_model(node_attr[row], node_attr[col])

        if not self.ignore_action:
            # Attach action to each state
            node_attr = torch.cat([node_attr, action_vec], dim=-1)

        node_attr = self._node_model(
            node_attr, edge_index, edge_attr)

        # [batch_size, num_nodes, hidden_dim]
        node_attr = node_attr.view(batch_size, num_nodes, -1)

        # we return the same thing as input but with a changed state
        # this allows us to stack GNNs
        return node_attr, action, viz

    def make_node_mlp_layers_(self, input_dim, output_dim, act_fn, layer_norm):

        layers = []

        for idx in range(self.num_layers):

            if idx == 0:
                # first layer, input_dim => hidden_dim
                layers.append(nn.Linear(input_dim, self.hidden_dim))
                layers.append(get_act_fn(act_fn))
            elif idx == self.num_layers - 2:
                # layer before the last, add layer norm
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                if layer_norm:
                    layers.append(nn.LayerNorm(self.hidden_dim))
                layers.append(get_act_fn(act_fn))
            elif idx == self.num_layers - 1:
                # last layer, hidden_dim => output_dim and no activation
                layers.append(nn.Linear(self.hidden_dim, output_dim))
            else:
                # all other layers, hidden_dim => hidden_dim
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(get_act_fn(act_fn))

        return layers
