from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
from tf_template import Coordinator

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


def get_coordinator(model_id):
    from .data_source import get_data_source
    from .inference_model import get_inference_model
    from .eval_model import get_eval_model
    from .train_model import get_train_model
    params = load_params(model_id)
    model_dir = os.path.join(models_dir, model_id)

    data_source = get_data_source(**params.get('data_source', {}))
    inference_model = get_inference_model(**params.get('inference_model', {}))
    eval_model = get_eval_model(**params.get('eval_model', {}))
    train_model = get_train_model(**params.get('train_model', {}))

    return Coordinator(
        data_source, inference_model, eval_model, train_model, model_dir)
