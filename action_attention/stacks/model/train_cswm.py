import copy as cp
import shutil
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data as torch_data
from ...stack import StackElement
from ...constants import Constants
from ...models.CSWM import CSWM
from ...models.CSWMSoftAttention import CSWMSoftAttention
from ...models.CSWMHardAttention import CSWMHardAttention
from ... import utils
from ...loaders.TransitionsDataset import TransitionsDataset
from ...loaders.PathDataset import PathDataset


class InitModel(StackElement):
    # Initialize either baseline C-SWM, C-SWM with hard attention or C-SWM with soft attention.
    def __init__(self, model_config, learning_rate, device, use_hard_attention, use_soft_attention, load_path=None):

        super().__init__()
        self.model_config = model_config
        self.learning_rate = learning_rate
        self.device = device
        self.use_hard_attention = use_hard_attention
        self.use_soft_attention = use_soft_attention
        self.load_path = load_path

        self.OUTPUT_KEYS = {Constants.MODEL, Constants.OPTIM}

    def run(self, bundle: dict, viz=False) -> dict:

        assert not (self.use_hard_attention and self.use_soft_attention)

        if self.use_hard_attention:
            model_cls = CSWMHardAttention
        elif self.use_soft_attention:
            model_cls = CSWMSoftAttention
        else:
            model_cls = CSWM

        model = model_cls(self.model_config, self.logger).to(self.device)
        model.apply(utils.weights_init)

        if self.load_path is not None:
            model.load_state_dict(torch.load(self.load_path, map_location=self.device))
            model.eval()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        return {Constants.MODEL: model, Constants.OPTIM: optimizer}


class InitTransitionsLoader(StackElement):
    # Initialize training loader.
    def __init__(self, root_path, batch_size, factored_actions=False):

        super().__init__()
        self.root_path = root_path
        self.batch_size = batch_size
        self.factored_actions = factored_actions
        self.OUTPUT_KEYS = {Constants.TRAIN_LOADER}

    def run(self, bundle: dict, viz=False) -> dict:

        dataset = TransitionsDataset(self.root_path, self.factored_actions)
        train_loader = torch_data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)
        return {Constants.TRAIN_LOADER: train_loader}


class InitPathLoader(StackElement):
    # Initialize evaluation loader.
    def __init__(self, root_path, path_length, batch_size, factored_actions=False):

        super().__init__()
        self.root_path = root_path
        self.path_length = path_length
        self.batch_size = batch_size
        self.factored_actions = factored_actions
        self.OUTPUT_KEYS = {Constants.EVAL_LOADER}

    def run(self, bundle: dict, viz=False) -> dict:

        dataset = PathDataset(self.root_path, self.path_length, self.factored_actions)
        eval_loader = torch_data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        return {Constants.EVAL_LOADER: eval_loader}


class Train(StackElement):

    def __init__(self, epochs, device, model_save_path):

        super(Train, self).__init__()

        self.epochs = epochs
        self.device = device
        self.model_save_path = model_save_path

        self.INPUT_KEYS = {Constants.MODEL, Constants.OPTIM, Constants.TRAIN_LOADER}
        self.OUTPUT_KEYS = {Constants.LOSSES}
        self.LOG_INTERVAL = 20

    def run(self, bundle: dict, viz=False) -> dict:

        model = bundle[Constants.MODEL]
        optimizer = bundle[Constants.OPTIM]
        train_loader = bundle[Constants.TRAIN_LOADER]

        self.logger.info('Starting model training...')
        step = 0
        best_loss = 1e9
        best_weights = None
        losses = []

        if self.model_save_path is not None:
            utils.maybe_create_dirs(utils.get_dir_name(self.model_save_path))

        for epoch in range(1, self.epochs + 1):

            model.train()
            train_loss = 0

            for batch_idx, data_batch in enumerate(train_loader):

                data_batch = [tensor.to(self.device) for tensor in data_batch]
                optimizer.zero_grad()

                if viz:
                    self.logger.info("Visualizing inputs.")
                    print(data_batch[1][0])
                    for i in range(5):
                        plt.subplot(2, 5, 1 + i)
                        plt.imshow(data_batch[0][0, i].cpu().numpy().transpose((1, 2, 0)))
                        plt.subplot(2, 5, 6 + i)
                        plt.imshow(data_batch[2][0, i].cpu().numpy().transpose((1, 2, 0)))
                    plt.show()

                loss = model.contrastive_loss(*data_batch)

                loss.backward()
                train_loss += loss.item()
                optimizer.step()

                losses.append(loss.item())

                if batch_idx % self.LOG_INTERVAL == 0:
                    self.logger.info(
                        "Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch, batch_idx * len(data_batch[0]),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                            loss.item() / len(data_batch[0])))

                step += 1

            avg_loss = train_loss / len(train_loader.dataset)
            self.logger.info('====> Epoch: {} Average loss: {:.6f}'.format(epoch, avg_loss))

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_weights = cp.deepcopy(model.state_dict())
                # save to an intermediate path
                # I want to distinguish between fully trained models and intermediate models
                if self.model_save_path is not None:
                    torch.save(model.state_dict(), utils.get_intermediate_save_path(self.model_save_path))

        if best_weights is not None:
            model.load_state_dict(best_weights)

        # move an intermediate model to the final save path
        if self.model_save_path is not None:
            shutil.move(utils.get_intermediate_save_path(self.model_save_path), self.model_save_path)

        return {Constants.LOSSES: losses}


class Eval(StackElement):

    HITS_AT = [1]

    def __init__(self, device, batch_size, num_steps, dedup):

        super(Eval, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.dedup = dedup

        self.INPUT_KEYS = {Constants.EVAL_LOADER, Constants.MODEL}
        self.OUTPUT_KEYS = {*[Constants.HITS.name + "_at_{:d}".format(k) for k in self.HITS_AT], Constants.MRR}

    def run(self, bundle: dict, viz=False) -> dict:

        self.logger.info("Evaluating {:d} step predictions.".format(self.num_steps))

        eval_loader = bundle[Constants.EVAL_LOADER]
        model = bundle[Constants.MODEL]
        model.eval()

        hits_at = defaultdict(int)
        num_samples = 0
        rr_sum = 0

        pred_states = []
        next_states = []
        next_ids = []

        with torch.no_grad():

            for batch_idx, data_batch in enumerate(eval_loader):

                data_batch = [[t.to(self.device) for t in tensor] for tensor in data_batch]

                if self.dedup:
                    observations, actions, state_ids = data_batch[:3]
                else:
                    observations, actions = data_batch[:2]

                if observations[0].size(0) != self.batch_size:
                    continue

                obs = observations[0]
                next_obs = observations[-1]
                if self.dedup:
                    next_id = state_ids[-1]

                state = model.forward(obs)
                next_state = model.forward(next_obs)

                pred_state = state
                for i in range(self.num_steps):
                    pred_state = model.forward_transition(pred_state, actions[i])

                if viz:
                    self.logger.info("Visualizing predictions.")
                    self.logger.info("### Note that it is up to the model to choose which object gets assigned to which slots. Hence, to interpret this plot, you need to compare the predicted attention weight with the object being moved. ###")

                    if model.use_attention:
                        weights = model.forward_weights(state, actions[0])

                    for i in range(obs.shape[0]):

                        plt.figure(figsize=(10, 5))

                        plt.subplot(2, 3, 1)
                        plt.title("current state")
                        if len(obs[i].shape) == 4:
                            plt.imshow(obs[i].cpu().numpy().sum(0).transpose((1, 2, 0)))
                        else:
                            plt.imshow(obs[i].cpu().numpy().transpose((1, 2, 0)))
                        plt.subplot(2, 3, 2)
                        plt.title("next state")
                        if len(next_obs[i].shape) == 4:
                            plt.imshow(next_obs[i].cpu().numpy().sum(0).transpose((1, 2, 0)))
                        else:
                            plt.imshow(next_obs[i].cpu().numpy().transpose((1, 2, 0)))

                        plt.subplot(2, 3, 3)
                        plt.title("predicted attention weights")
                        plt.bar(list(range(1, len(weights[i]) + 1)), weights[i].cpu().numpy())

                        plt.subplot(2, 3, 4)
                        plt.title("encoding of current state")
                        for j in range(5):
                            plt.scatter(
                                [state[i, j, 0].cpu().numpy()],
                                [state[i, j, 1].cpu().numpy()]
                            )
                        plt.subplot(2, 3, 5)
                        plt.title("encoding of next state")
                        for j in range(5):
                            plt.scatter(
                                [next_state[i, j, 0].cpu().numpy()],
                                [next_state[i, j, 1].cpu().numpy()]
                            )
                        plt.subplot(2, 3, 6)
                        plt.title("predicted next state")
                        for j in range(5):
                            plt.scatter(
                                [pred_state[i, j, 0].cpu().numpy()],
                                [pred_state[i, j, 1].cpu().numpy()]
                            )
                        plt.tight_layout()
                        plt.show()

                pred_states.append(pred_state.cpu())
                next_states.append(next_state.cpu())
                if self.dedup:
                    next_ids.append(next_id.cpu())

            pred_state_cat = torch.cat(pred_states, dim=0)
            next_state_cat = torch.cat(next_states, dim=0)
            if self.dedup:
                next_ids_cat = np.concatenate(next_ids, axis=0)

            if viz:
                self.logger.info("Visualizing all encodings.")
                for i in range(5):
                    plt.subplot(2, 5, 1 + i)
                    plt.scatter(
                        next_state_cat[:, i, 0].cpu().numpy().reshape(-1),
                        next_state_cat[:, i, 1].cpu().numpy().reshape(-1), label="obj {:d}".format(i)
                    )
                for i in range(5):
                    plt.subplot(2, 5, 6 + i)
                    plt.scatter(
                        pred_state_cat[:, i, 0].cpu().numpy().reshape(-1),
                        pred_state_cat[:, i, 1].cpu().numpy().reshape(-1), label="obj {:d}".format(i)
                    )
                plt.show()

            full_size = pred_state_cat.size(0)

            # Flatten object/feature dimensions
            next_state_flat = next_state_cat.view(full_size, -1)
            pred_state_flat = pred_state_cat.view(full_size, -1)

            dist_matrix = utils.pairwise_distance_matrix(next_state_flat, pred_state_flat)

            dist_matrix_diag = torch.diag(dist_matrix).unsqueeze(-1)
            dist_matrix_augmented = torch.cat(
                [dist_matrix_diag, dist_matrix], dim=1)

            # Workaround to get a stable sort in numpy.
            dist_np = dist_matrix_augmented.numpy()
            indices = []
            for row in dist_np:
                keys = (np.arange(len(row)), row)
                indices.append(np.lexsort(keys))
            indices = np.stack(indices, axis=0)

            if self.dedup:
                mask_mistakes = indices[:, 0] != 0
                closest_next_ids = next_ids_cat[indices[:, 0] - 1]

                if len(next_ids_cat.shape) == 2:
                    equal_mask = np.all(closest_next_ids == next_ids_cat, axis=1)
                else:
                    equal_mask = closest_next_ids == next_ids_cat

                indices[:, 0][np.logical_and(equal_mask, mask_mistakes)] = 0

            indices = torch.from_numpy(indices).long()

            labels = torch.zeros(
                indices.size(0), device=indices.device,
                dtype=torch.int64).unsqueeze(-1)

            num_samples += full_size

            for k in self.HITS_AT:
                match = (indices[:, :k] == labels).any(dim=1)
                num_matches = match.sum()
                hits_at[k] += num_matches.item()

            match = indices == labels
            _, ranks = match.max(1)

            reciprocal_ranks = torch.reciprocal(ranks.double() + 1)
            rr_sum += reciprocal_ranks.sum().item()

        res = dict()

        for k in self.HITS_AT:
            result_hits = hits_at[k] / float(num_samples)
            res[Constants.HITS.name + "_at_{:d}".format(k)] = result_hits
            self.logger.info("Hits @ {}: {}".format(k, result_hits))

        result_mrr = rr_sum / float(num_samples)
        res[Constants.MRR] = result_mrr
        self.logger.info("MRR: {}".format(result_mrr))

        return res
