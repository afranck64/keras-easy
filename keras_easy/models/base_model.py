import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import load_model
import numpy as np
import joblib

from .. import config as _config
from .. import util
from .. import scripters

from .tools import generators
from .tools import optimizers




class BaseModel(object):
    def __init__(self,
                 config:_config.Configuration,
                 class_weight=None,
                 nb_epoch=None,
                 batch_size=None,
                 patience=None,
                 freeze_layers_number=None,
                 metrics=None,
                 lr=None,
                 optimizer=None,
                 loss=None,
                 dropout=None):
        self.model = None
        self.class_weight = class_weight
        if config.main.classification_type == _config.CLASSIFICATION_TYPE.REGRESSION:
            self.class_mode = 'ordinal'
        else:
            self.class_mode = 'categorical'
        self.nb_epoch = nb_epoch if nb_epoch is not None else 10
        self.fine_tuning_patience = patience if patience is not None else 20
        self.batch_size = batch_size if batch_size is not None else 32
        self.freeze_layers_number = freeze_layers_number
        self.metrics = metrics if metrics is not None else ['mae']
        self.optimizer = optimizers.get_optimizer(optimizer, {'lr': lr})
        self.loss = loss if loss is not None else 'mean_squared_error'
        self.dropout = dropout if dropout is not None else 0.5
        self.img_size = (224, 224)
        self.run_config = config

    def _create(self):
        raise NotImplementedError('subclasses must override _create()')

    def _fine_tuning(self):
        self.freeze_top_layers()

        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics)
        self.model.summary()

        train_data = self.get_train_datagen(rotation_range=30., shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        validation_data = self.get_validation_datagen()
        callbacks = self.get_callbacks(self.run_config.get_fine_tuned_weights_path(), patience=self.fine_tuning_patience)

        if util.is_keras2():
            self.model.fit_generator(
                train_data,
                steps_per_epoch=len(train_data),
                epochs=self.nb_epoch,
                validation_data=validation_data,
                validation_steps=len(validation_data),
                callbacks=callbacks,
                class_weight=self.class_weight)
        else:
            self.model.fit_generator(
                train_data,
                samples_per_epoch=self.run_config.nb_train_samples,
                nb_epoch=self.nb_epoch,
                validation_data=validation_data,
                nb_val_samples=len(validation_data),
                callbacks=callbacks,
                class_weight=self.class_weight)

        self.model.save(self.run_config.get_model_path())

    def train(self):
        print("Creating model...")
        self._create()
        print("Model is created")
        print("Fine tuning...")
        if self.run_config.train.fine_tuning:
            weights_path = self.run_config.get_fine_tuned_weights_path()
            if os.path.exists(weights_path):
                if self.run_config.train.fine_tuning==_config.FINE_TUNING_TYPE.LOAD_BY_NAME:
                    self.model.load_weights(weights_path, by_name=True)
                else:
                    self.model.load_weights(weights_path)
        self._fine_tuning()
        self.save_classes()
        print("Classes are saved")

    def load(self):
        print("Creating model")
        #self.load_classes()
        self._create()
        self.model.load_weights(self.run_config.get_fine_tuned_weights_path(), by_name=True)
        #self.model = load_model(self.run_config.get_model_path())
        return self.model

    def save_classes(self):
        joblib.dump(self.run_config.classes, self.run_config.get_classes_path())

    def save(self):
        self.save_classes()
        self.model.save_weights(self.run_config.get_fine_tuned_weights_path())
        self.model.save(self.run_config.get_model_path())

    def get_input_tensor(self):
        if util.get_keras_backend_name() == 'theano':
            return Input(shape=(3,) + self.img_size)
        else:
            return Input(shape=self.img_size + (3,))

    @staticmethod
    def make_net_layers_non_trainable(model):
        for layer in model.layers:
            layer.trainable = False

    def freeze_top_layers(self):
        if self.freeze_layers_number:
            print("Freezing {} layers".format(self.freeze_layers_number))
            for layer in self.model.layers[:self.freeze_layers_number]:
                layer.trainable = False
            for layer in self.model.layers[self.freeze_layers_number:]:
                layer.trainable = True

    @staticmethod
    def get_callbacks(weights_path, patience=30, monitor='val_loss'):
        early_stopping = EarlyStopping(verbose=1, patience=patience, monitor=monitor)
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor=monitor, patience=max(int(patience/3), 5))
        model_checkpoint = ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True, monitor=monitor)
        return [early_stopping, reduce_lr_on_plateau, model_checkpoint]

    @staticmethod
    def apply_mean(image_data_generator):
        """Subtracts the dataset mean"""
        image_data_generator.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))

    def get_data_generator(self, *args, **kwargs):
        if not self.run_config.data.use_augmentation:
            args = ()
            kwargs = {}

        if self.run_config.main.scripter:
            if self.run_config.data.inmemory:
                idg = generators.MemDataGenerator(*args, row_processor=self.run_config.scripter.row_processor, **kwargs)
            else:
                idg = generators.DataGenerator(*args, row_processor=self.run_config.scripter.row_processor, **kwargs)
        else:
            #idg = generators.ImageDataGenerator(*args, **kwargs)
            idg = generators.DataGenerator(*args, row_processor=scripters.BaseScript.row_processor, **kwargs)
        return idg
    
    def load_classes(self):
        self.run_config.classes = joblib.load(self.run_config.get_classes_path())

    def load_img(self, img_path):
        img = image.load_img(img_path, target_size=self.img_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return preprocess_input(x)[0]

    def get_train_datagen(self, *args, **kwargs):
        idg = self.get_data_generator(*args, **kwargs)
        self.apply_mean(idg)
        if self.run_config.data.use_dir:
            return idg.flow_from_directory(self.run_config.data.train_dir, target_size=self.img_size, classes=self.run_config.classes, batch_size=self.batch_size, class_mode=self.class_mode)
        else:
            return idg.flow_from_file(self.run_config.data.train_file, target_size=self.img_size, classes=self.run_config.classes, batch_size=self.batch_size, class_mode=self.class_mode)

    def get_validation_datagen(self, *args, **kwargs):
        idg = self.get_data_generator(*args, **kwargs)
        self.apply_mean(idg)
        if self.run_config.data.use_dir:
            return idg.flow_from_directory(self.run_config.data.validation_dir, target_size=self.img_size, classes=self.run_config.classes, batch_size=self.batch_size, class_mode=self.class_mode)
        else:
            return idg.flow_from_file(self.run_config.data.validation_file, target_size=self.img_size, classes=self.run_config.classes, batch_size=self.batch_size, class_mode=self.class_mode)

    def get_test_datagen(self, path=None, *args, **kwargs):
        idg = self.get_data_generator()
        self.apply_mean(idg)
        if self.run_config.predict.accuracy:
            class_mode = self.class_mode
        else:
            #We don't expects the sample(s) at path to have ground truth values
            class_mode = None

        if self.run_config.data.use_dir:
            return idg.flow_from_directory(path or self.run_config.data.test_dir, target_size=self.img_size, classes=self.run_config.classes, batch_size=self.batch_size, class_mode=class_mode)
        else:
            return idg.flow_from_file(path or self.run_config.data.test_file, target_size=self.img_size, classes=self.run_config.classes, batch_size=self.batch_size, class_mode=class_mode)

    def __getattr__(self, name):
        if self.model:
            return getattr(self.model, name)
        else:
            raise AttributeError(name)