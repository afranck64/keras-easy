from keras.layers import (Input, Flatten, Dense, Dropout, AvgPool2D, GlobalAvgPool2D, MaxPool2D, Conv2D, BatchNormalization)
from keras.models import Model, Sequential

from .. import config as _config
from .base_model import BaseModel


class Mininet(BaseModel):
    noveltyDetectionLayerName = 'fc1'
    noveltyDetectionLayerSize = 2048

    def __init__(self, *args, **kwargs):        
        super(Mininet, self).__init__(*args, **kwargs)
        self.img_size = (224, 224)
        if not self.freeze_layers_number:
            # we chose to train the top 2 identity blocks and 1 convolution block
            self.freeze_layers_number = 0

    def _create(self):
        #self.make_net_layers_non_trainable(base_model)
        #input_shape = (self.run_config.train.batch_size, ) + self.img_size + (3,)
        x = Sequential()

        input_shape = self.img_size + (3,)
        x.add(Conv2D(64, (1, 1), strides=(1, 1), activation='relu', input_shape=input_shape))
        x.add(MaxPool2D((3, 3)))
        x.add(Conv2D(64, (1, 1), strides=(1, 1), activation='relu'))
        x.add(MaxPool2D((3, 3)))
        x.add(Conv2D(64, (1,1), strides=(1, 1), activation='relu'))
        x.add(MaxPool2D((3, 3)))
        x.add(Conv2D(64, (1,1), strides=(1, 1), activation='relu'))
        x.add(MaxPool2D((3, 3)))
        x.add(Conv2D(64, (1,1), strides=(1, 1), activation='relu'))
        x.add(BatchNormalization())
        x.add(Flatten())
        x.add(Dense(self.run_config.data.nb_classes * 2, activation='relu'))
        x.add(Dropout(self.run_config.model.dropout))

        # x = base_model.output
        # x = Dropout(self.dropout)(x)
        # x = Dropout(self.dropout)(x)
        # x = AvgPool2D()(x)
        # x = Flatten()(x)

        if self.run_config.main.classification_type==_config.CLASSIFICATION_TYPE.CLASSIFICATION:
            x.add(Dense(self.run_config.data.nb_classes, activation='softmax', name='predictions'))
        elif self.run_config.main.classification_type==_config.CLASSIFICATION_TYPE.REGRESSION:
            x.add(Dense(1, activation='linear', name='regression'))
        elif self.run_config.main.classification_type==_config.CLASSIFICATION_TYPE.MULTIPLE_REGRESSION:
            x.add(Dense(self.run_config.data.nb_classes, activation='linear', name='multiple_regression'))
        else:
            raise ValueError("Error, unknown classification_type: <%s>" % self.run_config.main.classification_type)

        #self.model = Model(input=x.input, output=predictions)
        self.model = x


def inst_class(*args, **kwargs):
    return Mininet(*args, **kwargs)
