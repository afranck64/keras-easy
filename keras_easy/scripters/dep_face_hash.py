import os
import glob
import warnings

import numpy as np
import pandas as pd
import keras
from keras import backend as K
from keras_preprocessing import image
import imagesize

from keras_easy.scripters import Script as _Script
from keras_easy import config


class Script(_Script):
    MUCT_PATH = "/mnt/0CC70FA40CC70FA4/workspace/image_analysis/muct/"
    HELEN_PATH = "/mnt/0CC70FA40CC70FA4/workspace/datasets/helen_faces/"
    USE_OPENCV_LANDMARKS = True
    IMAGE_SIZE = (480, 640)
    MAX_SCALE = 1000
    config = None
    @classmethod
    def row_processor(cls, row, image_data_generator, color_mode, target_size, interpolation, data_format, class_mode):
        filename = row['filename']
        img = image.load_img(filename,
                        color_mode=color_mode,
                        target_size=target_size,
                        interpolation=interpolation)
        x = image.img_to_array(img, data_format=data_format)
        # Pillow images should be closed after `load_img`,
        # but not PIL images.
        if hasattr(img, 'close'):
            img.close()
        params = image_data_generator.get_random_transform(x.shape)
        x = image_data_generator.apply_transform(x, params)
        x = image_data_generator.standardize(x)
        # build batch of labels
        if class_mode == 'input':
            y = x.copy()
        elif class_mode == 'sparse':
            y = np.array(row['class'].split(':'), dtype=K.floatx())
        elif class_mode == 'binary':
            y = np.array(row['class'].split(':'), dtype=K.floatx())
        elif class_mode == 'categorical':
            y = np.array(row['class'].split(':'), dtype=K.floatx())
            #y_val = transform_label_y_in_value(y)
            #y[y_val] = 1.0
        else:
            y = np.nan
        return x, y

    @classmethod
    def generate_labels(cls, run_config):
        #cls.generate_labels_muct(run_config)
        cls.generate_labels_helen(run_config)

    @classmethod
    def generate_labels_helen(cls, run_config):
        CACHE_FILENAME = os.path.join(run_config.data.data_dir, 'helen_labels_cache.csv')
        annotation_fnames = glob.glob(os.path.join(cls.HELEN_PATH, 'annotation', '*.txt'))
        images = glob.glob(os.path.join(cls.HELEN_PATH, 'helen_?', '*.jpg'))
        if os.path.exists(CACHE_FILENAME):
            warnings.warn("USING face_hash cached labels")
            df = pd.read_csv(CACHE_FILENAME)
        else:
            #Sorted:
            pass
            annotations_dict = {}
            annotations = []
            #contains  <image_name_without_extension:full_path_to_image>
            images_dict = {os.path.split(filename)[1].split('.')[0]:filename for filename in images}
            for fname_annotation in annotation_fnames:
                annot_df = pd.read_csv(fname_annotation)
                key = annot_df.columns[0]
                img_size = imagesize.get(images_dict[key])
                SCALE_SIZE = cls.MAX_SCALE/img_size[0], cls.MAX_SCALE/img_size[1]
                x,y = annot_df.index.values, annot_df[annot_df.columns[0]].values
                x *= SCALE_SIZE[0]
                y *= SCALE_SIZE[1]
                coords = np.empty(len(x)*2)
                for i in range(len(x)):
                    coords[2*i] = x[i]
                    coords[2*i+1] = y[i]
                coords_repr = ":".join("{:.2f}".format(coord) for coord in coords)
                annotations.append(
                    {
                        'id': key,
                        'filename': images_dict[key],
                        'class':coords_repr,
                    }
                )
            df = pd.DataFrame(annotations)           
            df.to_csv(CACHE_FILENAME, index=False)
        
        #
        nb_items = len(df)
        train_ratio = 1.0 - (run_config.split.validation_ratio + run_config.split.test_ratio)
        nb_train = int(nb_items * train_ratio)
        nb_validation = int(nb_items * run_config.split.validation_ratio)
        nb_test = nb_items - (nb_train + nb_validation)
        #pylint: disable=E0632
        train_indexes, validation_indexes, test_indexes = np.split(df.index, [nb_train, nb_train+nb_validation])
        train_df = df.loc[train_indexes]
        validation_df = df.loc[validation_indexes]
        test_df = df.loc[test_indexes]

        train_df.to_csv(run_config.data.train_file, index=False, float_format="%.3f")
        validation_df.to_csv(run_config.data.validation_file, index=False, float_format="%.3f")
        if run_config.data.test_file:
            test_df.to_csv(run_config.data.test_file, index=False, float_format="%.3f")




            #Generate a file cache

    @classmethod
    def generate_labels_muct(cls, run_config):
        CACHE_FILENAME = os.path.join(run_config.data.data_dir, 'labels_cache.csv')
        #We actually focus on the first camera: the <jpg> folder of muct
        #TODO: normalize the landmarks to [0...1] base on height and width
        if os.path.exists(CACHE_FILENAME):
            warnings.warn("USING face_hash cached labels")
            df = pd.read_csv(CACHE_FILENAME)
        else:
            #Sorted: 
            all_files = glob.glob(os.path.join(cls.MUCT_PATH, 'jpg*', '*.jpg'))
            all_keys = set(os.path.split(fname)[-1].split('.')[0] for fname in all_files)
            all_files.sort(key=lambda fname: os.path.split(fname)[-1])

            #TEST START
            #TEST END


            file_lst = glob.glob(os.path.join(cls.MUCT_PATH, 'jpg', '*.jpg'))

            fnames = {os.path.split(filename)[1][:-4] for filename in file_lst}
            if cls.USE_OPENCV_LANDMARKS:
                landmarks_fic = os.path.join(cls.MUCT_PATH, 'muct-landmarks', 'muct76-opencv.csv')
                SCALE_SIZE = cls.MAX_SCALE/cls.IMAGE_SIZE[0], cls.MAX_SCALE/cls.IMAGE_SIZE[1]
            else:
                landmarks_fic = os.path.join(cls.MUCT_PATH, 'muct-landmarks', 'muct76.csv')
                SCALE_SIZE = cls.MAX_SCALE*0.5/cls.IMAGE_SIZE[0], cls.MAX_SCALE*0.5/cls.IMAGE_SIZE[1]
            df = pd.read_csv(landmarks_fic)

            #Only work with frontal images : cameras a, d and e

            indexes = [True if not name.startswith('ir')  else False for name in df['name'].values]

            def _normalizer(row):
                for idx in range(2, 154,2):
                    row[idx] *= SCALE_SIZE[0]
                for idx in range(3, 154, 2):
                    row[idx] *= SCALE_SIZE[1]
                return row

            df = df.loc[indexes]
            col2str4 = lambda value: value[:4]
            copy_str = lambda value: str(value)
            #df['id'] = df['name'].apply(copy_str)
            #df['id']
            df['personid'] = df['name'].apply(col2str4)
            df.sort_values(by='name')
            id_column = df['name'].sort_values()
            df = df.apply(_normalizer, axis='columns')
            df['filename'] = all_files
            df = df.rename(index=str, columns={'name':'id'})
            row2str3 = lambda val: "%.3f" % val
            all2str3 = lambda arg: arg.apply(row2str3)
            df['class'] = df[df.columns[2:154]].apply(all2str3, axis='columns').apply(":".join, axis='columns')
    
            df = df[['filename', 'class', 'id', 'personid']]            
            df.to_csv(CACHE_FILENAME, index=False)


        # Only select images containing frontal faces
        mask = df['id'].apply(lambda name: bool(set('ade').intersection(set(name))))
        df = df[mask]

        new_index = np.arange(len(df))
        np.random.shuffle(new_index)
        df.index = new_index
        nb_items = len(df)
        train_ratio = 1.0 - (run_config.split.validation_ratio + run_config.split.test_ratio)
        nb_train = int(nb_items * train_ratio)
        nb_validation = int(nb_items * run_config.split.validation_ratio)
        nb_test = nb_items - (nb_train + nb_validation)
        #pylint: disable=E0632
        train_indexes, validation_indexes, test_indexes = np.split(df.index, [nb_train, nb_train+nb_validation])
        train_df = df.loc[train_indexes]
        validation_df = df.loc[validation_indexes]
        test_df = df.loc[test_indexes]

        train_df.to_csv(run_config.data.train_file, index=False, float_format="%.3f")
        validation_df.to_csv(run_config.data.validation_file, index=False, float_format="%.3f")
        if run_config.data.test_file:
            test_df.to_csv(run_config.data.test_file, index=False, float_format="%.3f")


if __name__ == "__main__":
    Script.generate_labels(config.Configuration.get_instance('../data/face_hash.cfg'))