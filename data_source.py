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

    def vis_inputs(self, mode=Modes.TRAIN, config=None):
        graph = tf.Graph()
        with graph.as_default():
            features, labels = self.get_inputs(mode, batch_size=None)

            session_creator = tf.train.ChiefSessionCreator(config=config)
            with tf.train.MonitoredSession(
                    session_creator=session_creator) as sess:
                while not sess.should_stop():
                    if labels is None:
                        vis = self.feature_vis(sess.run(features))
                    else:
                        fd, ld = sess.run((features, labels))
                        vis = get_vis(self.feature_vis(fd), self.label_vis(ld))
                    vis.show(block=False)
                    maybe_stop()
                    vis.close()
