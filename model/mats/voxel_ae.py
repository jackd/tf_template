from __future__ import division
import tensorflow as tf
from tf_template.model.builder import ModelBuilder
from parts import encode_voxels, decode
try:
    range = xrange
except NameError:
    # python3
    pass


def get_train_voxels():
    import numpy as np
    for i in range(int(1e7)):
        x = np.random.uniform(size=(32, 32, 32)) > 0.999
        yield x
    # example_ids = ['a', 'b', 'c']
    # for example_id in example_ids:
    #     yield load_voxels(example_id)


def disp_voxels(voxels):
    from mayavi import mlab
    import numpy as np
    mlab.figure()
    x, z, y = np.where(voxels)
    mlab.points3d(x, y, z)


def show():
    from mayavi import mlab
    mlab.show()


class VoxelAEBuilder(ModelBuilder):
    def get_inference(self, features, mode):
        """Get inferred value of the model."""
        return decode(encode_voxels(features, mode), mode)

    def get_inference_loss(self, inference, labels):
        """Get the loss assocaited with inferences."""
        return tf.nn.sigmoid_cross_entropy_with_logits(
            inference, tf.cast(labels, tf.float32))

    def get_train_op(self, loss, step):
        """
        Get the train operation.

        This operation is called within a `tf.control_dependencies(update_ops)`
        block, so implementations do not have to worry about update ops that
        are defined in the calculation of the loss, e.g batch_normalization
        update ops.
        """
        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        return optimizer.minimize(loss, global_step=step)

    def get_train_dataset(self):
        """
        Get a dataset giving features and labels for a single training example.

        Dataset must represent a 2-tuple, each of which can be a tensor, or
        possibly nested list/tuple/dict of tensors.
        """
        train_dataset = tf.data.Dataset.from_generator(
            get_train_voxels, output_types=tf.bool,
            output_shapes=(32, 32, 32))

        def map_fn(voxels):
            voxels = tf.cast(voxels, tf.float32)
            return voxels, voxels

        return train_dataset.map(map_fn, num_parallel_calls=8)

    def get_eval_dataset(self):
        """
        Get a dataset giving features and labels for a single eval example.

        Dataset must represent a 2-tuple, each of which can be a tensor, or
        possibly nested list/tuple/dict of tensors.
        """
        raise NotImplementedError('Abstract method')

    def get_predict_dataset(self):
        """
        Get the features for a single prediction example.

        Dataset must represent a tensor, or possibly nested list/tuple/dict of
        tensors.
        """
        raise NotImplementedError('Abstract method')

    def vis_example_data(self, feature_data, label_data):
        """
        Function for visualizing a batch of data for training or evaluation.

        All inputs are numpy arrays, or nested dicts/lists of numpy arrays.

        Not necessary for training/evaluation/infering, but handy for
        debugging.
        """
        for f, lab in zip(feature_data, label_data):
            disp_voxels(f)
            disp_voxels(lab)
            show()

    def vis_prediction_data(self, prediction_data, feature_data, label_data):
        """
        Function for visualizing a batch of data for training or evaluation.

        All inputs are numpy arrays, or nested dicts/lists of numpy arrays.

        `label_data` may be `None`.

        Not necessary for training/evaluation/infering, but handy for
        debugging.
        """
        raise NotImplementedError()

    def get_predictions(self, inferences):
        probs = tf.nn.sigmoid(inferences)
        predictions = tf.greater(inferences, 0)
        return dict(
            logits=inferences,
            probabilities=probs,
            predictions=predictions
        )

    def get_accuracy(self, predictions, labels):
        accuracy = tf.metrics.accuracy(labels, predictions['predictions'])
        tf.summary.scalar('accuracy', accuracy)
        return accuracy


def register_voxel_ae_family():
    from tf_template.model import register_builder_family
    register_builder_family('voxel_ae', VoxelAEBuilder)
