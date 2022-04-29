import copy as cp
import os
import pickle
import numpy as np
import gym
from skimage.io import imsave
from ...constants import Constants
from ...stack import StackElement
from ... import utils


class InitEnvAndSeed(StackElement):
    # Initialize the environment and set the random seed.
    def __init__(self, env_id, seed):

        super().__init__()
        self.env_id = env_id
        self.seed = seed
        self.OUTPUT_KEYS = {Constants.ENV}

    def run(self, bundle: dict, viz=False) -> dict:

        env = gym.make(self.env_id)

        np.random.seed(self.seed)
        env.action_space.seed(self.seed)
        env.seed(self.seed)

        return {
            Constants.ENV: env
        }


class CollectRandomAndSave(StackElement):
    # Collect data using a random policy. Save states as PNG images, and actions and positions as pickles.
    STATE_TEMPLATE = os.path.join("e_{:d}", "s_t_{:d}.png")
    ACTIONS_TEMPLATE = "actions.pkl"
    POSITIONS_TEMPLATE = "positions.pkl"

    def __init__(self, save_path, num_episodes, factored_actions=True):

        super().__init__()
        self.save_path = save_path
        self.num_episodes = num_episodes
        self.factored_actions = factored_actions
        self.INPUT_KEYS = {Constants.ENV}

    def run(self, bundle: dict, viz=False) -> dict:

        if os.path.isdir(self.save_path):
            raise ValueError("Save path already occupied.")

        env = bundle[Constants.ENV]
        actions = [[] for _ in range(self.num_episodes)]
        positions = [[] for _ in range(self.num_episodes)]

        for ep_idx in range(self.num_episodes):

            obs = env.reset()
            step_idx = 0
            self.save_image(
                ep_idx, step_idx,
                utils.float_0_1_image_to_uint8(obs[1])
            )
            positions[ep_idx].append(cp.deepcopy(obs[0]))

            while True:

                # select random action
                action = env.action_space.sample()
                obs, _, done, _ = env.step(action)

                actions[ep_idx].append(action)
                step_idx += 1
                self.save_image(
                    ep_idx, step_idx,
                    utils.float_0_1_image_to_uint8(obs[1])
                )
                positions[ep_idx].append(cp.deepcopy(obs[0]))

                if done:
                    break

            if ep_idx > 0 and ep_idx % 10 == 0:
                self.logger.info("episode {:d}".format(ep_idx))

        self.save_actions(actions)
        self.save_positions(positions)

        return {}

    def save_image(self, ep, step, img):

        save_path = os.path.join(self.save_path, self.STATE_TEMPLATE.format(ep, step))
        utils.maybe_create_dirs(utils.get_dir_name(save_path))
        imsave(save_path, img)

    def save_actions(self, actions):

        save_path = os.path.join(self.save_path, self.ACTIONS_TEMPLATE)
        utils.maybe_create_dirs(utils.get_dir_name(save_path))
        with open(save_path, "wb") as f:
            pickle.dump(actions, f)

    def save_positions(self, positions):

        save_path = os.path.join(self.save_path, self.POSITIONS_TEMPLATE)
        utils.maybe_create_dirs(utils.get_dir_name(save_path))
        with open(save_path, "wb") as f:
            pickle.dump(positions, f)
