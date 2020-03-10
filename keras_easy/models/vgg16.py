from keras.applications.vgg16 import VGG16 as KerasVGG16
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout

from .. import config as _config
from .base_model import BaseModel


class VGG16(BaseModel):
    noveltyDetectionLayerName = 'fc2'
    noveltyDetectionLayerSize = 4096

    def __init__(self, *args, **kwargs):
        super(VGG16, self).__init__(*args, **kwargs)

    def _create(self):
        base_model = KerasVGG16(weights='imagenet', include_top=False, input_tensor=self.get_input_tensor())
        self.make_net_layers_non_trainable(base_model)

        x = base_model.output
        x = Dropout(self.dropout)(x)
        x = Flatten()(x)
        #x = Dense(4096, activation='elu', name='fc1')(x)
        #x = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        #x = Dropout(0.6)(x)

        if self.run_config.main.classification_type==_config.CLASSIFICATION_TYPE.CLASSIFICATION:
            predictions = Dense(self.run_config.data.nb_classes, activation='softmax', name='predictions')(x)
        elif self.run_config.main.classification_type==_config.CLASSIFICATION_TYPE.REGRESSION:
            predictions = Dense(1, activation='linear', name='regression')(x)
        elif self.run_config.main.classification_type==_config.CLASSIFICATION_TYPE.MULTIPLE_REGRESSION:
            predictions = Dense(self.run_config.data.nb_classes, activation='linear', name='multiple_regression')(x)
        else:
            raise ValueError("Error, unknown classification_type: <%s>" % self.run_config.main.classification_type)

        self.model = Model(input=base_model.input, output=predictions)


def inst_class(*args, **kwargs):
    return VGG16(*args, **kwargs)
