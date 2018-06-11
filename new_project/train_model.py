from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_template import TrainModel


def get_train_model(
        batch_size=64, max_steps=10000,
        optimizer_key='adam', learning_rate=1e-3):
    from tf_template.deserialize import deserialize_optimization_op_fn
    raise NotImplementedError('TODO')
    # default implementation will work, but probably isn't optimal.
    return TrainModel(deserialize_optimization_op_fn(
        key=optimizer_key, learning_rate=learning_rate), batch_size, max_steps)
