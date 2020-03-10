import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from keras import backend as K
from keras.preprocessing.image import (ImageDataGenerator as _ImageDataGenerator, 
                                       DirectoryIterator as _DirectoryIterator,
                                       Iterator as _Iterator,
                                       load_img, img_to_array, array_to_img)
from keras.preprocessing import image


def _apply_func(args):
    func = args[0]
    args = args[1]
    return func(*args)

POOL = None

class CSVDataIterator(_Iterator):
    def __init__(self, labels_file, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 row_processor=None):
        if row_processor is None:
            #row_processor = lambda row, idg, color_mode, target_size, interpolation, data_format: (row['id'], row['class'])
            raise ValueError('row_processor should be callable function')
        self.row_processor = row_processor
        self.samples = pd.read_csv(labels_file)
        super().__init__(len(self.samples), batch_size, shuffle, seed)
        self.image_data_generator = image_data_generator
        self.row_processor = row_processor

        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'rgba', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "rgba", or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgba':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        #super().__init__('', *args, **kwargs)
        

    def _get_batches_of_transformed_samples(self, index_array):
        global POOL
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=K.floatx())
        batch_y = [None] * len(index_array)
        
        args = [(self.row_processor, (self.samples.loc[i], self.image_data_generator, self.color_mode, self.target_size, self.interpolation, self.data_format, self.class_mode)) for i in index_array]
        # build batch of image data
        #args = [self.row_processor, (self.samples.loc[j], self.image_data_generator, self.color_mode, self.target_size, self.interpolation, self.data_format, self.class_mode)]
        # if POOL is None:
        #     POOL = mp.Pool()
        # x_y = POOL.map(_apply_func, args)
        x_y = [_apply_func(arg) for arg in args]    ##Consume less memory than multi-processing
        for i, j in enumerate(index_array):
            #x, y = self.row_processor(self.samples.loc[j], self.image_data_generator, self.color_mode, self.target_size, self.interpolation, self.data_format, self.class_mode)
            # params = self.image_data_generator.get_random_transform(x.shape)
            # x = self.image_data_generator.apply_transform(x, params)
            # x = self.image_data_generator.standardize(x)
            x, y = x_y[i]
            batch_x[i] = x
            batch_y[i] = y

        batch_y = np.asarray(batch_y, dtype=K.floatx())
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        # if self.class_mode == 'input':
        #     batch_y = batch_x.copy()
        # elif self.class_mode == 'sparse':
        #     batch_y = np.asarray(raw_y)
        # elif self.class_mode == 'binary':
        #     batch_y = np.asarray(raw_y, dtype=K.floatx())
        # elif self.class_mode == 'categorical':
        #     batch_y = np.zeros(
        #         (len(batch_x), self.num_classes),
        #         dtype=K.floatx())
        #     for i, label in enumerate(raw_y[index_array]):
        #         batch_y[i, label] = 1.
        # else:
        #     return batch_x
        return batch_x, batch_y


    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


class DataGenerator(_ImageDataGenerator):
    def __init__(self, *args, row_processor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.row_processor = row_processor
        
    def flow_from_file(self, labels_file,
                            target_size=(256,256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):
        return CSVDataIterator(
            labels_file=labels_file, image_data_generator=self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation,
            row_processor=self.row_processor)

class CSVMemDataIterator(CSVDataIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_x = None
        self._cache_y = None

    def prefetch(self):
        #raise value
        raise NotImplementedError()
        self._cache_x, self._cache_y = super()._get_batches_of_transformed_samples(np.arange(len(self.samples)))

    def _get_batches_of_transformed_samples(self, index_array):
        return self._cache_x[index_array], self._cache_y[index_array]

class MemDataGenerator(DataGenerator):
    def __init__(self, *args, row_processor=None, **kwargs):
        super().__init__(*args, row_processor=row_processor, **kwargs)

    def flow_from_file(self,labels_file,
                            target_size=(256,256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):
        mem_iterator = CSVMemDataIterator(
            labels_file=labels_file, image_data_generator=self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation,
            row_processor=self.row_processor)
        #TODO: remove after test
        raise Exception("Should not prefetch -_-")
        mem_iterator.prefetch()
        return mem_iterator
    

class CSVImageIterator(_DirectoryIterator):
    """Iterator capable of reading images listed in a csv file.

    # Arguments
        labels_file: str
            path to a CSV file containing path to images <filename> and
            the corresponding label <class>            
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
            Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    """
    def __init__(self, labels_file, *args, **kwargs):
        df = pd.read_csv(labels_file)
        self.filenames = df['filename'].values
        self.classes = kwargs.get('classes') or df['class'].values
        super().__init__(*args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=K.floatx())
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = image.load_img(fname,
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = image.img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes),
                dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y



class ImageDataGenerator(_ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def flow_from_file(self, labels_file,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):
        return CSVImageIterator(
            labels_file, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)


if __name__ == "__main__":
    fic = "./../data/datasets/titanic/train.csv"
    dg = DataGenerator()
    it = dg.flow_from_file(fic)
    for i in it:
        print(it)