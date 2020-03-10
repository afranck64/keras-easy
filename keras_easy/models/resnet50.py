from keras.applications.resnet50 import ResNet50 as KerasResNet50
from keras.layers import (Flatten, Dense, Dropout)
from keras.models import Model

from ..import config as _config
from .base_model import BaseModel


class ResNet50(BaseModel):
    noveltyDetectionLayerName = 'fc1'
    noveltyDetectionLayerSize = 2048

    def __init__(self, *args, **kwargs):
        super(ResNet50, self).__init__(*args, **kwargs)
        if not self.freeze_layers_number:
            # we chose to train the top 2 identity blocks and 1 convolution block
            self.freeze_layers_number = 80

    def _create(self):
        base_model = KerasResNet50(include_top=False, input_tensor=self.get_input_tensor())
        self.make_net_layers_non_trainable(base_model)

        x = base_model.output
        x = Dropout(self.dropout)(x)
        x = Flatten()(x)
        # we could achieve almost the same accuracy without this layer, buy this one helps later
        # for novelty detection part and brings much more useful features.
        #x = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        #x = Dropout(0.5)(x)
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
    return ResNet50(*args, **kwargs)
