from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .visualization import get_vis
from .util import maybe_stop
ModeKeys = tf.estimator.ModeKeys


def get_dummy_input(spec):
    """Get random inputs based on the input `tf.layers.InputSpec`."""
    dtype = spec.dtype
    if dtype == tf.bool:
        return tf.random_normal(shape=spec.shape, dtype=tf.float32) > 0
    elif dtype == tf.string:
        shape = spec.shape
        dummy = tf.constant('dummy string')
        dummy = tf.reshape(dummy, (1,)*len(shape))
        return tf.tile(dummy, shape)
    else:
        return tf.random_uniform(
            shape=spec.shape, minval=dtype.min, maxval=dtype.max, dtype=dtype)


def get_dummy_inputs(*specs):
    """
    Get random inputs based on the (possibly nested) `tf.layers.InputSpec`s.
    """
    nest = tf.contrib.framework.nest
    return nest.map_structure(get_dummy_input, specs)


class DataSource(object):
    def get_inputs(self, mode, batch_size=None):
        """
        Get input tensors (or nested structures) for `features, labels`.

        Args:
            mode: one of tf.estimator.ModeKeys - 'train', 'eval', 'infer'
            batch_size: size of resulting batch

        Returns:
            (features, labels) each of which is a (possibly nested) structure
            of tensors.
        """
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
        """Get a vis for features as returned by get_inputs (first output)."""
        raise NotImplementedError('Abstract method')

    def label_vis(self, label):
        """Get a vis for labels as returned by get_inputs (second output)."""
        raise NotImplementedError('Abstract method')

    def input_vis(self, features, labels=None):
        """Get a vis for features and optionally labels."""
        vis = self.feature_vis(features)
        if labels is not None:
            vis = (vis, self.label_vis(labels))
        return vis

    def vis_input_data(self, features, labels=None):
        """Visualize (features, labels) associated with a single example."""
        vis = get_vis(self.input_vis(features, labels))
        vis.show(block=False)
        maybe_stop()
        vis.close()

    def get_epoch_length(self, mode):
        """Get the number of examples in the epoch."""
        return None

    def vis_inputs(self, mode=ModeKeys.TRAIN, config=None, batch_size=None):
        """Visualize inputs by building the graph and running the session."""
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

    def create_profile(
            self, path, mode, batch_size, skip_runs=10, config=None):
        from tf_toolbox.profile import create_profile
        create_profile(
            lambda: self.get_inputs(mode, batch_size), path,
            skip_runs=skip_runs, config=config)
