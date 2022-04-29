import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
from ...constants import Constants
from ...stack import StackElement


class MeasureSlotCorrelation(StackElement):
    # Measure correlation between object slots.
    def __init__(self, device):

        super().__init__()
        self.device = device
        self.INPUT_KEYS = {Constants.MODEL, Constants.EVAL_LOADER}
        self.OUTPUT_KEYS = {Constants.CORRELATION}

    @torch.no_grad()
    def run(self, bundle: dict, viz=False) -> dict:

        model = bundle[Constants.MODEL]
        eval_loader = bundle[Constants.EVAL_LOADER]
        model.eval()

        state_list = []

        for batch_idx, data_batch in enumerate(eval_loader):

            data_batch = [[t.to(self.device) for t in tensor] for tensor in data_batch]
            observations, actions = data_batch[:2]
            observations = torch.stack(observations, dim=0)
            s = observations.shape
            observations = observations.reshape((s[0] * s[1], s[2], s[3], s[4]))
            state = model.forward(observations)

            state_list.append(state.cpu())

        state_list = torch.cat(state_list, dim=0).numpy()
        num_objects = state_list.shape[1]

        if viz:
            for dim_idx in range(model.num_objects):
                for i in range(num_objects):
                    for j in range(num_objects):
                        plt.subplot(num_objects, num_objects, 1 + i + j * num_objects)
                        plt.scatter(state_list[:, i, dim_idx], state_list[:, j, dim_idx])
                plt.show()

        pearson_list = []
        for dim_idx in range(model.embedding_dim):
            for i in range(num_objects - 1):
                for j in range(i + 1, num_objects):
                    pearson = stats.pearsonr(state_list[:, i, dim_idx], state_list[:, j, dim_idx])[0]
                    pearson_list.append(pearson)
                    # self.logger.info("Pearson {:d}-{:d}, dim {:d}: {:f}".format(i, j, dim_idx, pearson))

        pearson_list = np.array(pearson_list, dtype=np.float32)
        abs_avg = np.mean(np.abs(pearson_list))
        self.logger.info("Average absolute Pearson: {:f}".format(abs_avg))
        return {
            Constants.CORRELATION: abs_avg
        }
