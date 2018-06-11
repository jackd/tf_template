from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


_optimizers = {
    'adam': tf.train.AdamOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
    'adadelta': tf.train.AdadeltaOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'gradientdescent': tf.train.GradientDescentOptimizer,
    'momentum': tf.train.MomentumOptimizer
}


def deserialize_optimizer(key, *args, **kwargs):
    return _optimizers[key.lower()](*args, **kwargs)


def deserialize_optimization_op_fn(key, *args, **kwargs):
    optimizer = deserialize_optimizer(key, *args, **kwargs)
    return optimizer.minimize
