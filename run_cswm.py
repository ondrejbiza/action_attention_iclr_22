from action_attention import utils
from action_attention.stack import Stack, Seeds, SacredLog, Sieve
from action_attention.stacks.model.train_cswm import InitModel, InitTransitionsLoader, InitPathLoader, Train, Eval
from action_attention.stacks.model.slot_correlation import MeasureSlotCorrelation
from action_attention import paths
from action_attention.constants import Constants

ex = utils.setup_experiment("config/cswm.json")
ex.add_config(paths.CFG_MODEL_CSWM)


@ex.capture()
def get_model_config(model_config):

    # turn variable names into constants
    d = {}
    utils.process_config_dict(model_config, d)
    return d


@ex.config
def config():

    seed = None
    use_hard_attention = False
    use_soft_attention = False
    device = "cuda:0"
    learning_rate = 5e-4
    batch_size = 1024
    epochs = 100
    model_save_path = None
    model_load_path = None
    dataset_path = "data/shapes_train"
    eval_dataset_path = "data/shapes_eval"
    viz_names = None


@ex.automain
def main(seed, use_hard_attention, use_soft_attention, device, learning_rate, batch_size, epochs, model_save_path,
         model_load_path, dataset_path, eval_dataset_path, viz_names):

    model_config = get_model_config()
    logger = utils.Logger()
    stack = Stack(logger)

    stack.register(Seeds(
        use_torch=True,
        device=device,
        seed=seed
    ))
    stack.register(InitModel(
        model_config=model_config,
        learning_rate=learning_rate,
        device=device,
        load_path=model_load_path,
        use_hard_attention=use_hard_attention,
        use_soft_attention=use_soft_attention
    ))

    if model_load_path is None:
        # train model
        stack.register(InitTransitionsLoader(
            root_path=dataset_path,
            batch_size=batch_size,
            factored_actions=False
        ))
        stack.register(Train(
            epochs=epochs,
            device=device,
            model_save_path=model_save_path
        ))
        stack.register(SacredLog(
            ex=ex,
            keys=[Constants.LOSSES],
            types=[SacredLog.TYPE_LIST]
        ))

    stack.register(Sieve(
        keys={Constants.MODEL}
    ))

    # evaluate model
    for i in [1, 5, 10]:

        stack.register(InitPathLoader(
            root_path=eval_dataset_path,
            path_length=i,
            batch_size=100,
            factored_actions=False
        ))
        stack.register(Eval(
            device=device,
            batch_size=100,
            num_steps=i,
            dedup=False
        ))
        keys = [*[Constants.HITS.name + "_at_{:d}".format(k) for k in Eval.HITS_AT], Constants.MRR]
        stack.register(SacredLog(
            ex=ex,
            keys=keys,
            types=[SacredLog.TYPE_SCALAR for _ in range(len(keys))],
            prefix="{:d}_step".format(i)
        ))
        stack.register(Sieve(
            keys={Constants.MODEL}
        ))

    # calculate correlation between slots
    stack.register(InitPathLoader(
        root_path=eval_dataset_path,
        path_length=10,
        batch_size=100,
        factored_actions=False
    ))
    stack.register(MeasureSlotCorrelation(
        device=device
    ))
    stack.register(SacredLog(
        ex=ex,
        keys=[Constants.CORRELATION],
        types=[SacredLog.TYPE_SCALAR]
    ))

    stack.forward(None, viz_names)
