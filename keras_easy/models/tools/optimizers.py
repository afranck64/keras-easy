from keras import optimizers as _optimizers


OPTIMIZERS = {
    "adadelta": _optimizers.adadelta,
    "adagrad": _optimizers.adagrad,
    "adam": _optimizers.Adam,
    "adamax": _optimizers.Adamax,
    "nadam": _optimizers.Nadam,
    "rmsprop": _optimizers.RMSprop,
    "sgd": _optimizers.SGD
}

OPTIMIZERS_PARAMS = {
}


def get_optimizer(optimizer, optimizer_params=None):
    optimizer = optimizer if optimizer is not None else "adam"
    # optimizer_params = optimizer_params if optimizer_params is not None else {}
    # optimizer_obj = OPTIMIZERS[optimizer](**optimizer_params)
    # return optimizer_obj
    return OPTIMIZERS[optimizer]()