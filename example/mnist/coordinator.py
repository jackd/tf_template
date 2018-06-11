from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
from tf_template import Coordinator, EvalModel, TrainModel
from tf_template.deserialize import deserialize_optimization_op_fn

_root_dir = os.path.realpath(os.path.dirname(__file__))
params_dir = os.path.join(_root_dir, 'params')
models_dir = os.path.join(_root_dir, '_models')
for d in (params_dir, models_dir):
    if not os.path.isdir(d):
        os.makedirs(d)


def load_params(model_id):
    params_path = os.path.join(params_dir, '%s.json' % model_id)
    if not os.path.isfile(params_path):
        raise ValueError('No params at %s' % params_path)
    with open(params_path, 'r') as fp:
        params = json.load(fp)
    return params


def get_mnist_coordinator(model_id):
    import tensorflow as tf
    from .data_source import MnistDataSource
    from .inference_model import MnistInferenceModel
    params = load_params(model_id)
    model_dir = os.path.join(models_dir, model_id)

    data_source = MnistDataSource(**params.get('data_source', {}))
    inference_model = MnistInferenceModel(**params.get('inference_model', {}))

    def inference_loss(inference, labels):
        return tf.losses.sparse_softmax_cross_entropy(
            logits=inference, labels=labels)

    def get_eval_metric_ops(predictions, labels):
        return dict(accuracy=tf.metrics.accuracy(labels, predictions['pred']))

    train_model_params = params.get('train_model', {})
    batch_size = train_model_params.get('batch_size', 64)
    max_steps = train_model_params.get('max_steps', 10000)
    optimizer_kwargs = train_model_params.get(
        'optimizer', {'key': 'adam', 'learning_rate': 1e-3})

    eval_model = EvalModel(inference_loss, get_eval_metric_ops)
    train_model = TrainModel(deserialize_optimization_op_fn(
        **optimizer_kwargs), batch_size, max_steps)

    return Coordinator(
        data_source, inference_model, eval_model, train_model, model_dir)
