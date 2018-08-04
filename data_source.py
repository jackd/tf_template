from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .visualization import get_vis
from .modes import Modes
from .util import maybe_stop


class DataSource(object):
    def get_inputs(self, mode, batch_size=None):
        raise NotImplementedError('Abstract method')

    def feature_vis(self, features):
        raise NotImplementedError('Abstract method')

    def label_vis(self, label):
        raise NotImplementedError('Abstract method')

    def input_vis(self, features, labels=None):
        vis = self.feature_vis(features)
        if labels is not None:
            vis = get_vis(vis, self.label_vis(labels))
        return vis

    def vis_input_data(self, features, labels=None):
        vis = self.input_vis(features, labels)
        vis.show(block=False)
        maybe_stop()
        vis.close()

    def vis_inputs(self, mode=Modes.TRAIN, config=None, batch_size=None):
        nest = tf.contrib.framework.nest
        graph = tf.Graph()
        with graph.as_default():
            features, labels = self.get_inputs(mode, batch_size=batch_size)
            tensors = dict(features=features)
            if labels is not None:
                tensors['labels'] = labels

            session_creator = tf.train.ChiefSessionCreator(config=config)
            with tf.train.MonitoredSession(
                    session_creator=session_creator) as sess:
                while not sess.should_stop():
                    record = sess.run(tensors)
                    if batch_size is None:
                        self.vis_input_data(**record)
                    else:
                        for ri in zip(*nest.flatten(record)):
                            self.vis_input_data(
                                **(nest.pack_sequence_as(tensors, ri)))
