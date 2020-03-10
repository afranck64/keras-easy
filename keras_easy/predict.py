import time
import argparse
import os
import numpy as np
import pandas as pd
import glob
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

from keras_easy import config
from keras_easy import util
from keras_easy import models


def get_parser(add_help=True):
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument('--path', dest='path', help='Path to image',type=str)
    parser.add_argument('--paths_file', default=None)
    parser.add_argument('config_file', help="Configuration file")
    parser.add_argument('--accuracy', type=int, help='To print accuracy score')
    parser.add_argument('--plot_confusion_matrix', type=int)
    parser.add_argument('--execution_time', type=int)
    parser.add_argument('--store_activations', type=int)
    # parser.add_argument('--novelty_detection', action='store_true')
    # parser.add_argument('--model', type=str, required=True, help='Base model architecture',
    #                     choices=[config.MODEL_RESNET50, config.MODEL_RESNET152, config.MODEL_INCEPTION_V3,
    #                              config.MODEL_VGG16, config.MODEL_MOBILENET])
    # parser.add_argument('--data_dir', help='Path to data train directory')
    parser.add_argument('--batch_size', default=500, type=int, help='How many files to predict on at once')
    return parser


def get_files(path):
    if os.path.isdir(path):
        files = glob.glob(path + '*.jpg')
        files += glob.glob(path + '*.png')
    elif path.find('*') > 0:
        files = glob.glob(path)
    elif path.lower().endswith('.csv') and os.path.exists(path):
        files = pd.read_csv(path)['filename'].sort_values()
        return files
    
    else:
        files = [path]

    if not len(files):
        print('No images found by the given path')
        exit(1)
    files.sort()
    return files


def get_inputs_and_trues(files):
    inputs = []
    y_true = []

    for i in files:
        x = model.load_img(i)
        try:
            image_class = i.split(os.sep)[-2]
            keras_class = int(classes_in_keras_format[image_class])
            y_true.append(keras_class)
        except Exception:
            y_true.append(os.path.split(i)[1])

        inputs.append(x)

    return inputs, y_true


# def predict(run_config, path):
#     cls_type = run_config.main.classification_type
#     #files = get_files(path)
#     n_files = len(files)
#     print('Found {} files'.format(n_files))

#     # if args.novelty_detection:
#     #     activation_function = util.get_activation_function(model, model.noveltyDetectionLayerName)
#     #     novelty_detection_clf = joblib.load(run_config.get_novelty_detection_model_path())

#     y_trues = []
#     predictions = [None] * n_files
#     nb_batch = int(np.ceil(n_files / float(run_config.predict.batch_size)))
#     #idg = model.get_test_datagen()
#     for n in range(0, nb_batch):
#         print('Batch {}'.format(n))
#         n_from = n * run_config.predict.batch_size
#         n_to = min(run_config.predict.batch_size * (n + 1), n_files)

#         #inputs, y_true = get_inputs_and_trues(files[n_from:n_to])
#         inputs, y_true = idg.next()
#         y_trues += y_true

#         # if args.store_activations:
#         #     util.save_activations(run_config, model, inputs, files[n_from:n_to], model.noveltyDetectionLayerName, n)

#         # if args.novelty_detection:
#         #     activations = util.get_activations(activation_function, [inputs[0]])
#         #     nd_preds = novelty_detection_clf.predict(activations)[0]
#         #     print(novelty_detection_clf.__classes[nd_preds])

#         if not run_config.predict.store_activations:
#             # Warm up the model
#             if n == 0:
#                 print('Warming up the model')
#                 start = time.clock()
#                 model.predict(np.array([inputs[0]]))
#                 end = time.clock()
#                 print('Warming up took {} s'.format(end - start))

#             # Make predictions
#             start = time.clock()
#             out = model.predict(np.array(inputs))
#             end = time.clock()
#             if cls_type==config.CLASSIFICATION_TYPE.MULTIPLE_REGRESSION:
#                 predictions[n_from:n_to] = out
#             else:
#                 predictions[n_from:n_to] = np.argmax(out, axis=1)
#             print('Prediction on batch {} took: {}'.format(n, end - start))

#     if not run_config.predict.store_activations:
#         print("PREDS: ", len(predictions), "trues: ", len(y_trues))
#         for i, p in enumerate(predictions):
#             if run_config.predict.plot_predictions:
#                 if run_config.predict.accuracy:
#                     util.plot_predictions(run_config, files[i], y_pred=predictions[i], y_true=y_trues[i])
#                 else:
#                     util.plot_predictions(run_config, files[i], y_pred=predictions[i], y_true=None)
#             recognized_class = list(classes_in_keras_format.keys())[list(classes_in_keras_format.values()).index(p)]
#             print('| should be {} ({}) -> predicted as {} ({})'.format(y_trues[i], files[i].split(os.sep)[-1], p,
#                                                                     recognized_class))

#         if run_config.predict.accuracy:
#             print('Accuracy {}'.format(accuracy_score(y_true=y_trues, y_pred=predictions)))

#         if run_config.predict.plot_confusion_matrix:
#             cnf_matrix = confusion_matrix(y_trues, predictions)
#             util.plot_confusion_matrix(cnf_matrix, run_config.classes, normalize=False)
#             util.plot_confusion_matrix(cnf_matrix, run_config.classes, normalize=True)




def predict(run_config, model, path):
    cls_type = run_config.main.classification_type
    files = get_files(path)
    n_files = 32    #TODO: fix later
    #print('Found {} files'.format(run_config.data))

    # if args.novelty_detection:
    #     activation_function = util.get_activation_function(model, model.noveltyDetectionLayerName)
    #     novelty_detection_clf = joblib.load(run_config.get_novelty_detection_model_path())

    y_trues = []
    X = []
    predictions = [None] * n_files
    #nb_batch = int(np.ceil(n_files / float(run_config.predict.batch_size)))

    print(run_config.data.test_file)
    #return
    idg = model.get_test_datagen(path)

    nb_batch = len(idg)
    for n in range(0, nb_batch):
        print('Batch {}'.format(n))
        n_from = n * run_config.predict.batch_size
        n_to = min(run_config.predict.batch_size * (n + 1), n_files)

        #inputs, y_true = get_inputs_and_trues(files[n_from:n_to])
        inputs, y_true = idg.next()
        if cls_type != config.CLASSIFICATION_TYPE.MULTIPLE_REGRESSION:
            y_true = np.argmax(y_true, axis=1)
        y_trues.extend(y_true)
        X.extend(inputs)
        
        # if args.store_activations:
        #     util.save_activations(run_config, model, inputs, files[n_from:n_to], model.noveltyDetectionLayerName, n)

        # if args.novelty_detection:
        #     activations = util.get_activations(activation_function, [inputs[0]])
        #     nd_preds = novelty_detection_clf.predict(activations)[0]
        #     print(novelty_detection_clf.__classes[nd_preds])

        if not run_config.predict.store_activations:
            # Warm up the model
            if n == 0:
                print('Warming up the model')
                start = time.clock()
                model.predict(np.array([inputs[0]]))
                end = time.clock()
                print('Warming up took {} s'.format(end - start))

            # Make predictions
            start = time.clock()
            out = model.predict(np.array(inputs))
            end = time.clock()
            if cls_type==config.CLASSIFICATION_TYPE.MULTIPLE_REGRESSION:
                predictions[n_from:n_to] = out
            else:
                predictions[n_from:n_to] = np.argmax(out, axis=1)
            print('Prediction on batch {} took: {}'.format(n, end - start))


    if not run_config.predict.store_activations:
        plot_prediction = True
        if plot_prediction:
            for i, p in enumerate(predictions):
                #print(predictions[i])
                if run_config.predict.plot_predictions:
                    if run_config.predict.accuracy:
                        util.plot_predictions(run_config, img_array=X[i], y_pred=predictions[i], y_true=y_trues[i])
                    else:
                        util.plot_predictions(run_config, img_array=X[i], y_pred=predictions[i], y_true=None)
                if cls_type != config.CLASSIFICATION_TYPE.MULTIPLE_REGRESSION:
                    #TODO: fix
                    recognized_class = list(classes_in_keras_format.keys())[list(classes_in_keras_format.values()).index(p)]
                    print('| should be {} ({}) -> predicted as {} ({})'.format(y_trues[i], files[i].split(os.sep)[-1], p,
                                                                        recognized_class))
        if not plot_prediction:    #save predictions
            out_filename = os.path.join(run_config.data.data_dir, 'predictions.csv')
            with open(out_filename, 'w') as outfile:
                outfile.write("filename,class\n")
                print(len(predictions), len(files))
                if cls_type == config.CLASSIFICATION_TYPE.MULTIPLE_REGRESSION:
                    for i,p in enumerate(predictions):
                        outfile.write(files[i] + "," + ":".join(p.astype(str)) + "\n")
                else:
                    for i,p in enumerate(predictions):
                        recognized_class = list(classes_in_keras_format.keys())[list(classes_in_keras_format.values()).index(p)]
                        outfile.write(files[i] + "," + recognized_class + "\n")



        if run_config.predict.accuracy:
            #print('Accuracy {}'.format(accuracy_score(y_true=y_trues, y_pred=predictions)))
            print('{Accuracy {}'.format((predictions==y_trues).sum()/len(predictions)))

            if run_config.predict.plot_confusion_matrix:
                cnf_matrix = confusion_matrix(y_trues, predictions)
                util.plot_confusion_matrix(cnf_matrix, run_config.classes, normalize=False)
                util.plot_confusion_matrix(cnf_matrix, run_config.classes, normalize=True)

def main(run_config):
    print('=' * 50)
    util.set_img_format()
    model = util.get_model_class_instance(run_config)
    model.load()
    classes_in_keras_format = util.get_classes_in_keras_format(run_config)
    data_path = None
    if run_config.data.use_dir:
        data_path = run_config.data.test_dir
    else:
        data_path = run_config.data.test_file
    predict(run_config, model, data_path)


def main_from_parser_args(parser_args):
    print('=' * 50)

    up_config = {
        'action.predict': {
            'batch_size': parser_args.batch_size,
            'accuracy': parser_args.accuracy,
            'plot_confusion_matrix': parser_args.plot_confusion_matrix,
        },
        'data': {
            'test_file': parser_args.path or parser_args.paths_file,
            'test_dir': parser_args.path or parser_args.paths_file
        }
    }

    if parser_args.config_file is not None:
        run_config = config.Configuration.get_instance(parser_args.config_file, up_config)
    else:
        #TODO: error
        run_config = config.Configuration.get_instance(up_config)
    
    main(run_config)


if __name__ == '__main__':
    tic = time.clock()
    parser_args = get_parser().parse_args()
    main_from_parser_args(parser_args)
