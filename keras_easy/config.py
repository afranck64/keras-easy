import os
import importlib
import pathlib
import configparser
from collections import namedtuple

from keras_easy import scripters


#ENUMS
CLASSIFICATION_TYPE = namedtuple("CLASSIFICATION_TYPE", ['CLASSIFICATION', 'REGRESSION', 'MULTIPLE_REGRESSION'])(-1, 0, 1)
FINE_TUNING_TYPE = namedtuple("FINE_TUNING_TYPE", ['LOAD_RANDOM', 'LOAD_BY_NAME', 'LOAD_ALL'])(0, 1, 2)

TrainConfig = namedtuple('TrainConfig', ['nb_epoch', 'batch_size', 'patience', 'fine_tuning'])
PredictConfig = namedtuple('PredictConfig', ['batch_size', 'accuracy', 'plot_confusion_matrix', 'store_activations',
                                             'plot_predictions'])

DataConfig = namedtuple('DataConfig', ['validation_dir', 'train_dir', 'test_dir', 'validation_file', 'train_file', 'test_file',
                                       'nb_classes', 'sorted_data_dir', 'data_dir',
                                       'use_dir', 'use_augmentation', 'inmemory', 'sorted_classes'])
"""DataConfig"""

MainConfig = namedtuple('MainConfig', ['name', 'action', 'scripter', 'classification_type', 'model', 'create_output_dirs'])
"""MainConfig:
name: str
    project name
action: str (train|predict|split)
    The intended action of the run
scripter: str
    Submodule in scripter which responsible for the class, it should have a Script class which extends base.Script
classification_type: int 
    -1: Run normal classification, 0: run regression with one final cell, 1: run regression with <nb_classes> outputs
create_output_dirs: bool
    If True, output directories will be created on at loading time"""
SplitConfig = namedtuple('SplitConfig', ['balanced', 'balancing_ratio', 'validation_ratio', 'test_ratio'])
ModelConfig = namedtuple('ModelConfig', ['name', 'lr', 'optimizer', 'dropout', 'momentum', 'decay', 'batch_size', 'freeze_layers_number',])
OutputDirectories = namedtuple('OutputDirectories', ['logs', 'trained'])

class Configuration(object):
    ABSPATH = pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent

    lock_file = os.path.join(ABSPATH, 'lock')

    #server_address = ('0.0.0.0', 4224)
    #buffer_size = 4096f

    _instance = None
    _config = None
    _config_file = None
    def __init__(self, main_cfg, data_cfg, train_cfg, predict_cfg, split_cfg, model_cfg, option_cfg):
        self._config:dict = None
        #self.model = 'mobilenet'
        self.classes = None
        self.nb_classes = 0
        self.main: MainConfig = MainConfig(**main_cfg)
        self.train: TrainConfig = TrainConfig(**train_cfg)
        self.predict: PredictConfig = PredictConfig(**predict_cfg)
        self.data: DataConfig = DataConfig(**data_cfg)
        self.model: ModelConfig = ModelConfig(**model_cfg)
        self.split: SplitConfig = SplitConfig(**split_cfg)
        self.output: OutputDirectories = OutputDirectories(*[os.path.join(self.data.data_dir, sub_dir) for sub_dir in OutputDirectories._fields])
        self.option: namedtuple = namedtuple('OptionConfig', (list(option_cfg)))(**option_cfg)
        self.scripter:scripters.Script
        self._scripter = None
        if self.main.create_output_dirs:
            for sub_dir in self.output:
                os.makedirs(sub_dir, exist_ok=True)
        if self.data.sorted_classes is not None:
            self.classes = self.data.sorted_classes.split(':')

        
        

    def get_top_model_weights_path(self):
        return os.path.join(self.output.trained, 'top-model-{}-weights.h5').format(self.main.model)


    def get_fine_tuned_weights_path(self, checkpoint=False):
        return os.path.join(self.output.trained, 'fine-tuned-{}-weights.h5').format(self.main.model + '-checkpoint' if checkpoint else self.main.model)


    def get_novelty_detection_model_path(self):
        return os.path.join(self.output.trained, 'novelty_detection-model-{}').format(self.main.model)


    def get_model_path(self):
        return os.path.join(self.output.trained, 'model-{}.h5').format(self.main.model)


    def get_classes_path(self):
        return os.path.join(self.output.trained, 'model-{}.h5').format(self.main.model)

    @property
    def scripter(self) -> scripters.Script:
        if self._scripter is None:
            try:
                self._scripter = importlib.import_module('keras_easy.scripters.' + self.main.scripter).Script
            except ModuleNotFoundError:
                self._scripter = importlib.import_module(self.main.scripter).Script
        return self._scripter

    @classmethod
    def get_instance(cls, config_file='../data/config.cfg', up_config=None):
        if cls._instance is None:
            cls._instance = cls.get_config(config_file, up_config)
        return cls._instance
    
    @classmethod
    def get_config(cls, config_file, up_config=None):
        _config = cls.load_config_dict(config_file, up_config=up_config)
        main_cfg = {key:None for key in MainConfig._fields}
        main_cfg.update(_config['main'])
        model_cfg = {key:None for key in ModelConfig._fields}
        model_cfg.update(_config.get('model.' + main_cfg['model'], _config.get('model.default')))
        model_cfg['name'] = model_cfg.get('name', main_cfg['model'])
        data_cfg = {key:None for key in DataConfig._fields}
        data_cfg.update(_config['data'])
        train_cfg = {key:None for key in TrainConfig._fields}
        train_cfg.update(_config['action.train'])
        predict_cfg = {key:None for key in PredictConfig._fields}
        predict_cfg.update(_config['action.predict'])
        split_cfg = {key:None for key in SplitConfig._fields}
        split_cfg.update(_config['action.split'])
        option_cfg = _config.get('option', {})
        run_config = Configuration(
            main_cfg=main_cfg,
            data_cfg=data_cfg,
            train_cfg=train_cfg,
            predict_cfg=predict_cfg,
            split_cfg=split_cfg,
            model_cfg=model_cfg,
            option_cfg=option_cfg
        )
        return run_config

    @classmethod
    def load_config_dict(cls, config_file='../data/config.cfg', up_config=None):
        _config = configparser.ConfigParser()
        _config.read(config_file)
        up_config = up_config or {}
        up_config = {key:{sub_key:sub_value for sub_key,sub_value in value.items() if sub_value is not None} for key,value in up_config.items()}

        ### Brute force type conversion
        _dict_config = {}
        for section in _config:
            _dict_config[section] = {}
            for option in _config[section]:
                cast_methods = [_config[section].getint, _config[section].getfloat, _config[section].getboolean, _config[section].get]
                for get_casted in cast_methods:
                    try:
                        casted_value = get_casted(option)
                        if casted_value == "":
                            casted_value = None
                        _dict_config[section][option] = casted_value
                        break
                    except ValueError:
                        pass

        all_sections = _dict_config
        for section in all_sections:
            if '.' in section:
                base, part = section.split('.')
                if part=='default':
                    for g_section in all_sections:
                        if '.' in g_section and base==g_section.split('.')[0]:
                            for key, value in all_sections[section].items():
                                if key not in all_sections[g_section]:
                                    all_sections[g_section][key] = value
        #_config.model = _config.get('action.'+_config.get('main','action'),'model')

        all_sections_keys = set(all_sections)
        for key in up_config:
            # if key  in all_sections:
            #     all_sections[key].update(up_config[key])
            if key not in all_sections_keys:
                if "." in key:
                    base, part = key.split('.')
                    for key2 in all_sections_keys:
                        if key2.startswith(base) and key2.endswith('.default'):
                            all_sections[key] = all_sections[key2].copy()
                else:
                    assert 'Key unavailable?'
            all_sections[key].update(up_config[key])

        return _dict_config



if __name__ == "__main__":
    cfg = Configuration.get_instance()
    print(cfg._config)