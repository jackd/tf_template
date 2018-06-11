from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tf_template import InferenceModel


class NewProjectInferenceModel(InferenceModel):
    def __init__(self, **kwargs):
        raise NotImplementedError('TODO')

    def get_inference(self, features, mode):
        raise NotImplementedError('TODO')

    def prediction_vis(self, prediction_data):
        raise NotImplementedError('TODO')


def get_inference_model(**kwargs):
    return NewProjectInferenceModel(**kwargs)
