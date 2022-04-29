import pickle
import os
import numpy as np
from skimage.io import imread
from .Dataset import Dataset


class PathDataset(Dataset):
    # This data loader is used during evaluation.
    STATE_TEMPLATE = os.path.join("e_{:d}", "s_t_{:d}.png")
    ACTIONS_TEMPLATE = "actions.pkl"
    POSITIONS_TEMPLATE = "positions.pkl"

    def __init__(self, root_path, path_length, factored_actions=True):

        super().__init__()
        self.root_path = root_path
        self.path_length = path_length
        self.factored_actions = factored_actions
        self.actions = None
        self.positions = None

        self.load_actions()
        self.load_positions()
        self.num_steps = len(self.actions)

    def __getitem__(self, ep):

        obs = []
        actions = []

        for step in range(self.path_length):

            obs.append(self.preprocess_image(self.load_image(ep, step)))
            if not self.factored_actions:
                action = self.obj_action_to_pos_action(ep, step)
            else:
                action = self.actions[ep][step]
            actions.append(action)

        obs.append(self.preprocess_image(self.load_image(ep, self.path_length)))

        return obs, actions

    def load_actions(self):

        load_path = os.path.join(self.root_path, self.ACTIONS_TEMPLATE)
        if not os.path.isfile(load_path):
            raise ValueError("Actions not found.")
        with open(load_path, "rb") as f:
            self.actions = pickle.load(f)

    def load_positions(self):

        load_path = os.path.join(self.root_path, self.POSITIONS_TEMPLATE)
        if not os.path.isfile(load_path):
            raise ValueError("Positions not found.")
        with open(load_path, "rb") as f:
            self.positions = pickle.load(f)

    def load_image(self, ep, step):

        load_path = os.path.join(self.root_path, self.STATE_TEMPLATE.format(ep, step))
        return imread(load_path)

    def obj_action_to_pos_action(self, ep, step):

        action = self.actions[ep][step]
        obj_idx = action // 4
        dir_idx = action % 4

        position = self.positions[ep][step][obj_idx]
        new_action = np.zeros(5 + 5 + 4, dtype=np.float32)
        new_action[position[0]] = 1
        new_action[position[1] + 5] = 1
        new_action[dir_idx + 10] = 1
        return new_action
