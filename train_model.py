from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


class TrainModel(object):
    def __init__(self, batch_size, max_steps):
        self._batch_size = batch_size
        self._max_steps = max_steps

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_steps(self):
        return self._max_steps

    def get_inference_loss(self, inference, labels):
        raise NotImplementedError('Abstract method')

    def get_total_loss(self, inference, labels):
        inference_loss = self.get_inference_loss(inference, labels)
        losses = tf.get_collection(tf.GraphKeys.LOSSES)
        if len(losses) == 0:
            return inference_loss
        else:
            if inference_loss not in losses:
                losses.append(inference_loss)
            return tf.add_n(losses)

    def get_optimization_op(self, loss, global_step):
        raise NotImplementedError('Abstract method')

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

    @staticmethod
    def from_fns(inference_loss_fn, optimization_op_fn, batch_size, max_steps):
        return TrainModelBase(
            inference_loss_fn, optimization_op_fn, batch_size, max_steps)


class TrainModelBase(TrainModel):
    def __init__(self, inference_loss_fn, optimization_op_fn, batch_size,
                 max_steps):
        self._inference_loss_fn = inference_loss_fn
        self._optimization_op_fn = optimization_op_fn
        super(TrainModelBase, self).__init__(batch_size, max_steps)

    def get_inference_loss(self, inference, labels):
        return self._inference_loss_fn(inference, labels)

    def get_optimization_op(self, loss, global_step):
        return self._optimization_op_fn(loss, global_step)
