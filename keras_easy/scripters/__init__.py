import abc

class Script(object):

    @abc.abstractclassmethod
    def row_processor(cls, row, image_data_generator, color_mode, target_size, interpolation, data_format, class_mode): #TODO: missing an argument for label2value
        raise NotImplementedError()

    @abc.abstractclassmethod
    def generate_labels(cls, run_config):
        """Generate label files,
        produce two csv files: train_file and validation_file,
        saved respectively at run_config.data.train_file and run_config.data.validation_file"""
        raise NotImplementedError()
    
    @abc.abstractclassmethod
    def plot_predictions(cls, run_config):
        pass


class BaseScript(Script):
    @classmethod
    def row_processor(cls, row, image_data_generator, color_mode, target_size, interpolation, data_format, class_mode): #TODO: missing an argument for label2value)
        pass
    #Pour 19h, jene suis pas du quartier)