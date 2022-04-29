import h5py
import collections
import sys
import os
import logging
import time
from threading import Thread, Event
from queue import Queue
from threading import Lock
import numpy as np
import torch
from torch import nn
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from incense import ExperimentLoader
from pvectorc import PVector
from . import constants
from .constants import Constants


def log_list(name, items, ex):

    for item in items:
        ex.log_scalar(name, item)


def get_dir_name(path):

    return os.path.dirname(path)


def maybe_create_dirs(dir_path):

    if len(dir_path) == 0:
        return

    if not os.path.isdir(dir_path):
        if os.path.isfile(dir_path):
            raise ValueError("File of the same name as target directory found.")
        os.makedirs(dir_path)


def setup_experiment(name):

    ex = Experiment(name)
    if constants.MONGO_URI is not None and constants.DB_NAME is not None:
        print("MongoDB observer.")
        ex.observers.append(MongoObserver(url=constants.MONGO_URI, db_name=constants.DB_NAME))
    else:
        print("FileDB observer.")
        maybe_create_dirs("data")
        ex.observers.append(FileStorageObserver("data/file_db"))

    return ex


def delete_query(loader, query):

    res_db = loader.find(query)
    num_deleted = 0

    for exp in res_db:
        num_deleted += 1
        exp.delete(confirmed=True)

    return num_deleted


def float_0_1_image_to_uint8(image):

    assert (image.dtype in [np.float32, np.float64]) and (image.max() <= 1.)
    return (image * 255).astype(np.uint8)


def image_chw_to_hwc(image):

    return image.transpose((1, 2, 0))


class Logger:

    def __init__(self, save_file=None, print_logs=True):

        self.save_file = save_file
        self.print_logs = print_logs

        self.logger = logging.getLogger("my_logger")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers = []

        if self.save_file is not None:

            file_dir = os.path.dirname(self.save_file)

            if len(file_dir) > 0 and not os.path.isdir(file_dir):
                os.makedirs(file_dir)

            file_handler = logging.FileHandler(self.save_file)
            file_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(file_handler)

        if self.print_logs:

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def close(self):

        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)


def get_act_fn(act_fn):
    # https://github.com/tkipf/c-swm
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for `act_fn`.')


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


def process_config_dict(kwargs, d):

    for key, value in kwargs.items():

        if type(value) == dict or type(value) == sacred.config.custom_containers.ReadOnlyDict:
            # a nested config dictionary
            d[Constants(key.upper())] = {}
            process_config_dict(value, d[Constants(key.upper())])
        else:
            # turn a variable into a constant
            # e.g. embedding_dim => Constants.EMBEDDING_DIM
            d[Constants(key.upper())] = value


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def get_intermediate_save_path(model_save_path):
    return model_save_path + "_"


def pairwise_distance_matrix(x, y):

    num_samples = x.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1).expand(num_samples, num_samples, dim)
    y = y.unsqueeze(0).expand(num_samples, num_samples, dim)

    return torch.pow(x - y, 2).sum(2)


def setup_mock_executor(gpu_list, jobs_per_gpu):
    return MockExecutor(gpu_list, jobs_per_gpu)


def check_jobs_done_mock(jobs, executor):

    while True:

        num_finished = sum(executor.check_job_done(job) for job in jobs)

        if num_finished == len(jobs):
            break

        time.sleep(5)


class MockExecutor:

    def __init__(self, gpu_list, jobs_per_gpu):

        self.gpu_list = gpu_list
        self.jobs_per_gpu = jobs_per_gpu

        # fc_queue is for jobs to run
        self.fc_queue = Queue()
        # gpu_queue is for available gpus
        self.gpu_queue = Queue()

        # done dict indicates which jobs are done
        self.done_dict = dict()
        self.done_dict_lock = Lock()

        # running list keeps track of running jobs
        self.running_list = list()
        self.running_list_lock = Lock()

        # each job gets an index
        self.running_job_idx = 0

        # enqueue available gpus and start worker threads
        self.enqueue_gpus_()
        self.worker_run_thread, self.worker_release_thread = None, None
        self.worker_release_thread_flag = Event()
        self.run_threads_()

    def submit(self, run_fc, *args, **kwargs):

        job_idx = self.running_job_idx
        self.running_job_idx += 1

        self.done_dict_lock.acquire()
        self.done_dict[job_idx] = False
        self.done_dict_lock.release()

        self.fc_queue.put((run_fc, job_idx, args, kwargs))

        return job_idx

    def check_job_done(self, job_idx):

        self.done_dict_lock.acquire()
        done = self.done_dict[job_idx]
        self.done_dict_lock.release()

        return done

    def stop(self):

        self.fc_queue.put(None)
        self.worker_release_thread_flag.set()

        self.worker_run_thread.join()
        self.worker_release_thread.join()

    def enqueue_gpus_(self):

        for gpu in self.gpu_list:
            for _ in range(self.jobs_per_gpu):
                self.gpu_queue.put(gpu)

    def run_threads_(self):

        self.worker_run_thread = Thread(target=self.worker_run_)
        self.worker_run_thread.start()

        self.worker_release_thread = Thread(target=self.worker_release_)
        self.worker_release_thread.start()

    def worker_run_(self):

        while True:

            item = self.fc_queue.get()

            if item is None:
                break
            else:
                run_fc, job_idx, args, kwargs = item

            gpu_idx = self.gpu_queue.get()

            kwargs["gpu"] = gpu_idx
            process = run_fc(*args, **kwargs)

            self.running_list_lock.acquire()
            self.running_list.append((job_idx, gpu_idx, process))
            self.running_list_lock.release()

    def worker_release_(self):

        while True:

            self.running_list_lock.acquire()

            if self.worker_release_thread_flag.is_set() and len(self.running_list) == 0:
                self.running_list_lock.release()
                break

            to_delete = []

            for idx, item in enumerate(self.running_list):

                job_idx, gpu_idx, process = item

                if process.poll() is not None:

                    self.done_dict_lock.acquire()
                    self.done_dict[job_idx] = True
                    self.done_dict_lock.release()

                    self.gpu_queue.put(gpu_idx)

                    to_delete.append(idx)

            for idx in reversed(to_delete):

                del self.running_list[idx]

            self.running_list_lock.release()

            # there's no queue so better sleep
            time.sleep(5)


def get_experiment_loader():

    return ExperimentLoader(
        mongo_uri=constants.MONGO_URI,
        db_name=constants.DB_NAME
    )


def execute_query(loader, query, config_keys, metric_keys):

    res = collections.defaultdict(list)
    res_db = loader.find(query)

    for exp in res_db:

        flag = False
        for metric_key in metric_keys:
            if metric_key not in exp.metrics:
                flag = True
                break

        if flag:
            continue

        config_values = []
        for key in config_keys:

            key = key.split(".")
            value = exp.config

            for part in key:
                value = getattr(value, part)

            if isinstance(value, PVector):
                value = str(value.tolist())

            config_values.append(value)

        config_values = tuple(config_values)
        res[config_values].append([exp.metrics[key].values for key in metric_keys])

    return res


def load_list_dict_h5py(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            array_dict.append(dict())
            for key in hf[grp].keys():
                array_dict[i][key] = hf[grp][key][:]
    return array_dict
