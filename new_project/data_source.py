from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_template import DataSource


class NewProjectDataSource(DataSource):
    def __init__(self, **kwargs):
        # TODO
        raise NotImplementedError('TODO')

    def get_inputs(self, mode, batch_size=None):
        raise NotImplementedError('Abstract method')

    def feature_vis(self, features):
        raise NotImplementedError('Abstract method')

    def label_vis(self, label):
        raise NotImplementedError('Abstract method')


def get_data_source(**kwargs):
    return NewProjectDataSource(**kwargs)
