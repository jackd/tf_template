from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


class TrainModel(object):
    """
    Object responsible for training a model.

    Non-abstract implementations must implement:
        get_inference_loss
        get_optimization_op
    """
    def __init__(self, batch_size, max_steps):
        self._batch_size = batch_size
        self._max_steps = max_steps

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_steps(self):
        return self._max_steps

    # def get_total_loss_from_inference_loss(self, inference_loss):
    #     tf.summary.scalar('inference_loss', inference_loss)
    #     losses = [inference_loss]
    #     for key in (tf.GraphKeys.REGULARIZATION_LOSSES, tf.GraphKeys.LOSSES):
    #         losses.extend(tf.get_collection(key))
    #     # Ensure we don't double count
    #     losses = list(set(losses))

    #     if len(losses) == 1:
    #         return inference_loss
    #     else:
    #         return tf.add_n(losses)

    def get_regularization_loss(self):
        losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(losses) == 0:
            return None
        losses = list(set(losses))
        if len(losses) == 1:
            return losses[0]
        else:
            return tf.add_n(losses)

    def combine_losses(self, inference_loss):
        reg_loss = self.get_regularization_loss()
        if reg_loss is None:
            return inference_loss
        else:
            tf.summary.scalar('reg_loss', reg_loss)
            tf.summary.scalar('inference_loss', inference_loss)
            return inference_loss + reg_loss

    def get_total_loss(self, inference, labels):
        inference_loss = self.get_inference_loss(inference, labels)
        return self.combine_losses(inference_loss)

    def get_inference_loss(self, inference, labels):
        raise NotImplementedError('Abstract method')

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
            return tf.group(*update_ops)

    @staticmethod
    def from_fns(
            inference_loss_fn, optimization_op_fn, batch_size, max_steps):
        return TrainModelBase(
            inference_loss_fn, optimization_op_fn, batch_size, max_steps)

    def to_dict(self):
        return dict(
            inference_loss_fn=self.get_inference_loss,
            optimization_op_fn=self.get_optimization_op,
            batch_size=self.batch_size,
            max_steps=self.max_steps
        )

    def rebuild(self, **kwargs):
        kw = self.to_dict()
        kw.update(kwargs)
        return self._rebuild(**kw)

    def _rebuild(self, **kwargs):
        return TrainModel.from_fns(**kwargs)


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


class UpdateOpsRunner(TrainModel):
    """
    TrainModel Implementation that just run update ops.

    e.g. useful for settling batch norm statistics after training.
    """
    def get_inference_loss(self, inference, labels):
        raise RuntimeError('should not be here')

    def get_total_loss(self, inference, labels):
        return tf.zeros((), dtype=tf.float32)

    def get_optimization_op(self, loss, global_step):
        raise RuntimeError('should not be here')

    def get_train_op(self, loss):
        global_step = tf.train.get_or_create_global_step()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            inc = global_step.assign_add(1)
        return inc
