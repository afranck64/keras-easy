import argparse


import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras_easy import (split, train, predict)



def get_parser(add_help=True):
    parser = argparse.ArgumentParser(prog='python -m keras_easy', add_help=add_help)
    sub_parsers = parser.add_subparsers(dest='command')
    sub_parsers.add_parser('split', parents=[split.get_parser(add_help=False)])
    sub_parsers.add_parser('train', parents=[train.get_parser(add_help=False)])
    sub_parsers.add_parser('predict', parents=[predict.get_parser(add_help=False)])
    return parser

if __name__ == "__main__":
    parser_args = get_parser().parse_args()
    print("ARGS: ", parser_args)
    if parser_args.command == 'split':
        split.main_from_parser_args(parser_args)
        #print("Done!!!")
    elif parser_args.command == 'train':
        train.main_from_parser_args(parser_args)
    elif parser_args.command == 'predict':
        predict.main_from_parser_args(parser_args)