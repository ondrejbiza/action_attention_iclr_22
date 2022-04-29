from enum import Enum


constants = ["CNN_HIDDEN_DIM", "MLP_HIDDEN_DIM", "GNN_HIDDEN_DIM", "EMBEDDING_DIM", "ACTION_DIM", "NUM_OBJECTS",
             "HINGE", "SIGMA", "IGNORE_ACTION", "COPY_ACTION", "NUM_GNN_LAYERS", "ENCODER", "ATTENTION",
             "ENV", "MODEL", "CORRELATION", "OPTIM", "TRAIN_LOADER", "EVAL_LOADER", "LOSSES", "VALUE_SIZE",
             "HITS", "MRR"]

# all constants must be unique
assert len(constants) == len(set(constants))

Constants = Enum("Constants", {
    c: c for c in constants
})

MONGO_URI = None