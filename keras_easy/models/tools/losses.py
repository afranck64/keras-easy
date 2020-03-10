import keras.backend as K

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


#TODO: make use of <config> and config.data.nb_classes
def get_loss(loss):
    if loss=="mse" or loss=="mean_squared_error":
        return mse
    if rmse=="rmse" or loss=="root_mean_squared_error":
        return rmse
    return loss