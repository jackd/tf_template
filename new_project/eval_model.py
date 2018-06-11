from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_template.eval_model import EvalModel


def get_eval_model(**kwargs):
    raise NotImplementedError('TODO')
    # default implementation might work, but probably isn't optimal.

    def inference_loss(inference, labels):
        return tf.losses.sparse_softmax_cross_entropy(
            logits=inference, labels=labels)

    def get_eval_metric_ops(predictions, labels):
        return dict(accuracy=tf.metrics.accuracy(labels, predictions['pred']))

    return EvalModel(inference_loss, get_eval_metric_ops)
