from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class EvalModel(object):
    def __init__(self, inference_loss_fn, eval_metric_ops_fn=None):
        self._inference_loss_fn = inference_loss_fn
        self._eval_metric_ops_fn = eval_metric_ops_fn

    def get_inference_loss(self, inference, labels):
        return self._inference_loss_fn(inference, labels)

    def get_total_loss(self, inference, labels):
        inference_loss = self.get_inference_loss(inference, labels)
        losses = tf.get_collection(tf.GraphKeys.LOSSES)
        if len(losses) == 0:
            return inference_loss
        else:
            if inference_loss not in losses:
                losses.append(inference_loss)
            return tf.add_n(losses)

    def get_eval_metric_ops(self, predictions, labels):
        if self._eval_metric_ops_fn is None:
            return None
        else:
            return self._eval_metric_ops_fn(predictions, labels)
