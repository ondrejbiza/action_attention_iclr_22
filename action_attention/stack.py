import signal
import numpy as np
import torch
from action_attention import utils


class StackElement:

    def __init__(self, logger=None, pass_through=True, has_mock=False):

        self.logger = logger
        self.pass_through = pass_through
        self.has_mock = has_mock

        self.INPUT_KEYS = set()
        self.OUTPUT_KEYS = set()
        self.CLOSE_FC = None

    def run(self, bundle: dict, viz=False) -> dict:

        raise NotImplementedError()

    def __call__(self, bundle: dict) -> dict:

        return self.forward(bundle)

    def forward(self, bundle: dict, mock=False, viz=False) -> dict:

        if mock:
            bundle = self.mock_input()

        self.validate_bundle_(bundle, self.INPUT_KEYS)

        output_bundle = self.run(bundle, viz=viz)

        self.validate_bundle_(output_bundle, self.OUTPUT_KEYS)

        if self.pass_through:
            # the inputs and outputs shouldn't overlap, that's what pass through is for
            if len(set(bundle.keys()).intersection(set(output_bundle.keys()))) != 0:
                self.logger.error("Input and output overlap in pass-through: {:s}.".format(self.__class__.__name__))
                self.logger.error("Input: {:s}".format(str(bundle.keys())))
                self.logger.error("Output: {:s}".format(str(output_bundle.keys())))

                # delete the overlapping items from the input bundle
                intersection = set(bundle.keys()).intersection(set(output_bundle.keys()))
                for item in intersection:
                    del bundle[item]

                #raise ValueError()
            return {**bundle, **output_bundle}
        else:
            return output_bundle

    def forward_viz(self, bundle: dict, mock=False) -> dict:

        signal.signal(signal.SIGINT, signal.default_int_handler)

        output_bundle = self.forward(bundle, mock, viz=True)

        # either the run method will use the viz flag
        # or there will be a viz_output method
        if hasattr(self, "viz_output") and self.viz_output is not None:
            # usually I'd have a plotting loop over the entire dataset
            # this allows me to exit from that without killing the program
            try:
                self.viz_output(output_bundle)
            except KeyboardInterrupt:
                pass

        return output_bundle

    def validate_bundle_(self, bundle: dict, keys: set) -> None:

        bundle_keys = set(bundle.keys())
        keys = set(keys)

        if keys.issubset(bundle_keys.intersection(keys)):
            return
        else:
            self.logger.error("Error: input/output keys do not match specification: {:s}.".format(
                self.__class__.__name__
            ))
            self.logger.error("Expected keys: {}".format(str(keys)))
            self.logger.error("Received keys: {}".format(str(bundle_keys)))
            raise ValueError("Error: input/output keys do not match specification.")


class Sieve(StackElement):

    def __init__(self, keys: set):
        super().__init__(None, pass_through=False)

        self.INPUT_KEYS = keys
        self.OUTPUT_KEYS = keys

    def run(self, bundle: dict, viz=False) -> dict:

        new_bundle = {}
        for key in bundle.keys():
            if key in self.INPUT_KEYS:
                new_bundle[key] = bundle[key]

        return new_bundle


class SacredLog(StackElement):

    TYPE_SCALAR = 0
    TYPE_LIST = 1

    def __init__(self, ex, keys: list, types: list, prefix=None, logger=None):
        super().__init__(logger=logger)

        assert len(keys) == len(types)
        self.ex = ex
        self.keys = keys
        self.types = types
        self.prefix = prefix

        self.INPUT_KEYS = set(keys)

    def run(self, bundle: dict, viz=False) -> dict:

        for key, val_type in zip(self.INPUT_KEYS, self.types):

            if isinstance(key, str):
                name = key
            else:
                name = key.name

            if self.prefix is not None:
                name = "{:s}_{:s}".format(self.prefix, name)

            self.logger.info("Saving {:s} to DB.".format(name))

            if val_type == self.TYPE_SCALAR:
                self.ex.log_scalar(name, bundle[key])
            elif val_type == self.TYPE_LIST:
                utils.log_list(name, bundle[key], self.ex)
            else:
                raise ValueError("Invalid value type.")

        return {}


class Seeds(StackElement):

    def __init__(self, use_torch, device, seed):
        super().__init__(None, pass_through=True)
        self.use_torch = use_torch
        self.device = device
        self.seed = seed

    def run(self, bundle: dict, viz=False) -> dict:

        np.random.seed(self.seed)

        if self.use_torch:
            torch.manual_seed(self.seed)
            if self.device != "cpu":
                torch.cuda.manual_seed(self.seed)

        return {}


class Stack:

    def __init__(self, logger):

        self.logger = logger
        self.elements = []

    def register(self, element: StackElement):

        self.elements.append(element)

    def print_stack_(self):

        self.logger.info("Printing stack:")

        for element in self.elements:
            self.logger.info("{:s}".format(element.__class__.__name__))

    def forward(self, mock_names=None, viz_names=None) -> dict:

        self.set_loggers_()
        self.print_stack_()

        bundle = {}
        for element in self.elements:
            mock = (mock_names is not None) and (element.__class__.__name__ in mock_names)
            viz = (viz_names is not None) and (element.__class__.__name__ in viz_names)

            if mock:
                assert element.has_mock

            if viz:
                bundle = element.forward_viz(bundle, mock=mock)
            else:
                bundle = element.forward(bundle, mock=mock)

        # close
        for element in self.elements:
            if element.CLOSE_FC is not None:
                element.CLOSE_FC()

        return bundle

    def set_loggers_(self):

        for element in self.elements:
            element.logger = self.logger
