from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .visualization import get_vis
from .modes import Modes
from .util import maybe_stop


def get_dummy_input(spec):
    dtype = spec.dtype
    return tf.random_uniform(
        shape=spec.shape, minval=dtype.min, maxval=dtype.max, dtype=dtype)


def get_dummy_inputs(*specs):
    nest = tf.contrib.framework.nest
    return nest.map_structure(get_dummy_input, specs)


class DataSource(object):
    def get_inputs(self, mode, batch_size=None):
        raise NotImplementedError('Abstract method')

    def get_input_spec(self, mode, batch_size=None):
        """
        Get a (possibly nested) `tf.layers.InputSpec` this source produces.

        If `batch_size` is `None`, the returned structure will be associated
        with no batching.

        Args:
            `mode`: one of `tf.estimator.ModeKeys`
            `batch_size`: size of the batch, or None if no batching is to be
                applied.
        Returns:
            (feature_spec, label_spec) tuple, where each is either a possible-
                nested `tf.layers.InputSpec` or `None`.
        """
        raise NotImplementedError('Abstract method')

    def get_dummy_inputs(self, mode, batch_size=None):
        """
        Get random inputs with the same shape/dtype/structure as inputs.

        Useful for profiling, as it can show whether or not a choke-point is
        occuring in the DataSource.
        """
        return get_dummy_inputs(*self.get_input_spec(mode, batch_size))

    def feature_vis(self, features):
        raise NotImplementedError('Abstract method')

    def label_vis(self, label):
        raise NotImplementedError('Abstract method')

    def input_vis(self, features, labels=None):
        vis = self.feature_vis(features)
        if labels is not None:
            vis = (vis, self.label_vis(labels))
        return vis

    def vis_input_data(self, features, labels=None):
        vis = get_vis(self.input_vis(features, labels))
        vis.show(block=False)
        maybe_stop()
        vis.close()

    def vis_inputs(self, mode=Modes.TRAIN, config=None):
        graph = tf.Graph()
        with graph.as_default():
            features, labels = self.get_inputs(mode, batch_size=None)
            tensors = dict(features=features)
            if labels is not None:
                tensors['labels'] = labels

            session_creator = tf.train.ChiefSessionCreator(config=config)
            with tf.train.MonitoredSession(
                    session_creator=session_creator) as sess:
                while not sess.should_stop():
                    record = sess.run(tensors)
                    self.vis_input_data(**record)
