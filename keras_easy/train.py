import numpy as np
import argparse
import traceback
import os

#np.random.seed(1337)  # for reproducibility

from keras_easy import config
from keras_easy import util


def get_parser(add_help=True):
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument('config_file', help='Path to config file', default='../data/config.cfg')
    # parser.add_argument('--data_dir', help='Path to data dir')
    # parser.add_argument('--model', type=str, help='Base model architecture', choices=[
    #     config.MODEL_RESNET50,
    #     config.MODEL_RESNET152,
    #     config.MODEL_INCEPTION_V3,
    #     config.MODEL_VGG16,
    #     config.MODEL_MOBILENET])
    parser.add_argument('--fine_tuning', type=int)
    parser.add_argument('--nb_epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--freeze_layers_number', type=int, help='will freeze the first N layers and unfreeze the rest')
    return parser


def init(run_config:config.Configuration):
    util.lock(run_config)
    util.set_img_format()
    util.override_keras_directory_iterator_next()
    util.set_classes(run_config)
    util.set_samples_info(run_config)

    if util.get_keras_backend_name() != 'theano':
        util.tf_allow_growth()


def main(run_config):
    model = util.get_model_class_instance(
        run_config,
        class_weight=None, #util.get_class_weight(run_config.data.train_dir), #TODO: check later
        nb_epoch=run_config.train.nb_epoch,
        batch_size=run_config.model.batch_size,
        patience=run_config.train.patience,
        freeze_layers_number=run_config.model.freeze_layers_number,
        dropout=run_config.model.dropout,
        lr=run_config.model.lr)
    try:
        model.train()
        print('Training is finished!')
    except KeyboardInterrupt:
        model.save()
        print("Training stopped by the user")
        print("Model saved.")
    except Exception as e:
        model.save()
        raise e

def main_from_parser_args(parse_args):
    up_config = {
        'action.train': {
            'nb_epoch': parse_args.nb_epoch,
            'batch_size': parse_args.batch_size,
            'fine_tuning': parse_args.fine_tuning
        },
        'model.default': {
            'freeze_layers_number': parse_args.freeze_layers_number
        }
    }

    run_config = config.Configuration.get_instance(parse_args.config_file, up_config)

    main(run_config)

if __name__ == '__main__':
    parser_args = get_parser().parse_args()
    main_from_parser_args(parser_args)