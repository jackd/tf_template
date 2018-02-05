import os

params_dir = os.path.join(os.path.dirname(__file__), 'params')


def get_params_path(model_id):
    return os.path.join(params_dir, '%s.json' % model_id)


def load_params(model_id):
    import json
    path = get_params_path(model_id)
    if not os.path.isfile(path):
        raise ValueError('No parameter file found at %s for model %s' %
                         (path, model_id))
    with open(path, 'r') as fp:
        params = json.load(fp)
    return params


_family_builders = {}


def register_builder_family(family_key, builder_fn):
    """
    Register the provided builder_fn under the given key.

    Args:
        family_key: hashable, generally a string
        builder_fn: a function with signature
            (model_id, params) => ModelBuilder
    """
    if family_key in _family_builders:
        raise KeyError('family_key %s already registered' % family_key)
    _family_builders[family_key] = builder_fn


def get_builder(model_id):
    params = load_params(model_id)
    family = params['family']
    return _family_builders[family](model_id, params)
