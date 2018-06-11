from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tf_template.visualization import ImageVis, PrintVis
from tf_template.data_source import DataSource

_data_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), '_data')
if not os.path.isdir(_data_dir):
    os.makedirs(_data_dir)


# def get_mnist_dataset(mode):
#     from tensorflow.examples.tutorials.mnist import input_data
#     data = input_data.read_data_sets(_data_dir, one_hot=False)
#     if mode == 'train':
#         data = data.train
#     elif mode == 'eval':
#         data = data.validation
#     elif mode in ('test', 'infer'):
#         data = data.test
#     else:
#         raise ValueError('Unrecognized mode "%s"' % mode)
#     images = data.images
#     labels = data.labels
#     return tf.data.Datset.from_tensor_slices((images, labels))


def get_mnist_dataset(mode):
    import official.mnist.dataset as ds
    # may have to make the following change to official.mnist.dataset:
    ## url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'  # NOQA
    # url = 'http://yann.lecun.com/exdb/mnist/' + filename + '.gz'
    # mode = 'train'
    if mode == 'train':
        return ds.train(_data_dir)
    elif mode in ('eval', 'infer'):
        return ds.test(_data_dir)
    else:
        raise ValueError('Unrecognized mode "%s"' % mode)


class MnistDataSource(DataSource):
    def __init__(self, corruption_stddev=None, shuffle_buffer=10000):
        self._corruption_stddev = corruption_stddev
        self._shuffle_buffer = shuffle_buffer

    def get_inputs(self, mode, batch_size=None):
        dataset = get_mnist_dataset(mode)

        if mode == 'train':
            dataset = dataset.repeat().shuffle(self._shuffle_buffer)
            stddev = self._corruption_stddev
            if stddev is not None:
                def map_fn(image, labels):
                    image += tf.random_normal(
                        stddev=stddev, shape=tf.shape(image), dtype=tf.float32)
                    return image, labels
                dataset = dataset.map(map_fn)

        if batch_size is not None:
            dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        image, labels = dataset.make_one_shot_iterator().get_next()
        shape = (28, 28, 1)
        if batch_size is not None:
            shape = (-1,) + shape
        image = tf.reshape(image, shape)
        return image, labels

    def feature_vis(self, features):
        return ImageVis(np.squeeze(features, axis=-1))

    def label_vis(self, labels):
        return PrintVis(labels)
