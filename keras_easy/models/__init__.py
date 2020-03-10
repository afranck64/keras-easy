import os
import glob
import importlib

_MODELS_DIR = os.path.split(__file__)[0]

def _get_models():
    non_model_modules = {'__init__', 'base_model'}
    potential_model_modules = set(os.path.splitext(filename)[0] for filename in glob.glob1(_MODELS_DIR, '*.py'))
    model_modules = potential_model_modules - non_model_modules
    return model_modules

MODELS = _get_models()

def get_model(run_config, *args, **kwargs):
    if run_config.main.model in MODELS:
        module_name = "keras_easy.models.{}".format(run_config.main.model)
    else:
        module_name = run_config.main.model
    
    model_module = importlib.import_module(module_name)
    return model_module.inst_class(run_config, *args, **kwargs)

