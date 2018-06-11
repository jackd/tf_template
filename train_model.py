from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


class TrainModel(object):
    def __init__(self, optimization_op_fn, batch_size, max_steps):
        self._optimization_op_fn = optimization_op_fn
        self._batch_size = batch_size
        self._max_steps = max_steps

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_steps(self):
        return self._max_steps

    def get_optimization_op(self, loss, global_step):
        return self._optimization_op_fn(loss, global_step)

    def get_train_op(self, loss):
        global_step = tf.train.get_or_create_global_step()
        opt_op = self.get_optimization_op(loss, global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if len(update_ops) == 0:
            return opt_op
        else:
            if opt_op not in update_ops:
                update_ops.append(opt_op)
            return tf.group(update_ops)
