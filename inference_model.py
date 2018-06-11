from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class InferenceModel(object):
    def get_inference(self, features, mode):
        raise NotImplementedError('Abstract method')

    def get_predictions(self, features, inference):
        return inference

    def prediction_vis(self, prediction_data):
        raise NotImplementedError('Abstract method')

    def needs_initial_weight_transfer(self):
        return False

    def load_initial_weights(self, features):
        pass
