from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_template import InferenceModel
from tf_template.visualization import PrintVis


class MnistInferenceModel(InferenceModel):
    def __init__(
            self, conv_filters=[16,  32, 64], dense_units=[32], n_classes=10):
        self.conv_filters = conv_filters
        self.dense_units = dense_units
        self.n_classes = n_classes

    def get_inference(self, features, mode):
        training = mode == tf.estimator.ModeKeys.TRAIN
        with tf.variable_scope('inference'):
            x = features
            for f in self.conv_filters:
                x = tf.layers.conv2d(x, f, 3, 1, padding='SAME')
                x = tf.layers.batch_normalization(
                    x, scale=False, training=training)
                x = tf.nn.relu(x)
                x = tf.layers.max_pooling2d(x, 3, 2, padding='SAME')
            x = tf.reduce_mean(x, axis=(1, 2))
            for u in self.dense_units:
                x = tf.layers.dense(x, u)
                x = tf.layers.batch_normalization(
                    x, scale=False, training=training)
                x = tf.nn.relu(x)
            x = tf.layers.dense(x, self.n_classes)
        return x

    def get_predictions(self, features, inference):
        probs = tf.nn.softmax(inference, axis=-1)
        preds = tf.argmax(inference, axis=-1)
        return dict(probs=probs, pred=preds)

    def prediction_vis(self, prediction_data):
        return PrintVis('%d: %s' % (
            prediction_data['pred'], str(prediction_data['probs'])))
