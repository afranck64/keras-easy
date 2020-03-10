import importlib
import argparse

from keras_easy import config

def get_parser(add_help=True):
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument('config_file', help="Configuration file")
    return parser

def main(run_config):
    run_config.scripter.generate_labels(run_config)

def main_from_parser_args(parser_args):
    #args = get_parser().parse_args()
    split_cfg = {'main': {'action':'split'}}
    run_config:config.Configuration = config.Configuration.get_instance(parser_args.config_file, split_cfg)
    main(run_config)



if __name__ == "__main__":
    args = get_parser().parse_args()
    split_cfg = {'main': {'action':'split'}}
    run_config:config.Configuration = config.Configuration.get_instance(args.config_file, split_cfg)
    run_config.scripter.generate_labels(run_config)