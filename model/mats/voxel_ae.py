from __future__ import division
import numpy as np
import tensorflow as tf
from tf_template.model.builder import ModelBuilder


def disp_voxels(voxels, **kwargs):
    from shapenet.mayavi_vis import vis_voxels
    from mayavi import mlab
    mlab.figure()
    vis_voxels(voxels, **kwargs)


def show():
    from mayavi import mlab
    mlab.show()


class VoxelAEBuilder(ModelBuilder):
    def encode_voxels(self, voxels, mode):
        # training = mode == tf.estimator.ModeKeys.TRAIN
        encoder_params = self.params.get('encoder', {})
        with tf.variable_scope('encoder'):
            x = tf.expand_dims(voxels, axis=-1)
            sizes = encoder_params.get('sizes', [7, 5, 3, 3])
            n_filters = encoder_params.get('n_filters', [16, 32, 16, 8])
            for size, n in zip(sizes, n_filters):
                x = tf.layers.conv3d(
                    x, n, size, 1, activation=tf.keras.layers.PReLU(),
                    padding='VALID')
                # x = tf.layers.batch_normalization(
                #     x, scale=False, training=training)
                # x = tf.layers.max_pooling3d(x, 3, 2)
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 64, activation=tf.keras.layers.PReLU())
        return x

    def decode(self, latent, mode):
        n_final = 6
        decoder_params = self.params.get('decoder', {})
        with tf.variable_scope('deocder'):
            x = latent
            initial_depth = decoder_params.get('initial_depth', 8)
            x = tf.layers.dense(
                x, initial_depth*n_final**3,
                activation=tf.keras.layers.PReLU())
            x = tf.reshape(x, (-1,) + (n_final,)*3 + (initial_depth,))

            sizes = decoder_params.get('sizes', [3, 3, 5, 7])
            n_filters = decoder_params.get('n_filters', [8, 16, 32, 8])

            for size, n in zip(sizes, n_filters):
                x = tf.layers.conv3d_transpose(
                    x, n, (size,)*3, 1, activation=tf.keras.layers.PReLU(),
                    padding='VALID')
            x = tf.layers.conv3d_transpose(
                x, 1, 1, 1, activation=None, padding='VALID')
            x = tf.squeeze(x, axis=-1)
        return x

    def get_inference(self, features, mode):
        """Get inferred value of the model."""
        return self.decode(self.encode_voxels(features, mode), mode)

    def get_inference_loss(self, inference, labels):
        """Get the loss assocaited with inferences."""
        loss_type = self.params.get('loss_type', 'x-entropy')
        if loss_type == 'x-entropy':
            if labels.dtype != tf.float32:
                labels = tf.cast(labels, tf.float32)
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=inference, labels=labels))
        elif loss_type == 'mse':
            return tf.reduce_mean((tf.sigmoid(inference) - labels)**2)
        else:
            raise KeyError('Unrecognized `loss_type` %s' % loss_type)

    def get_train_op(self, loss, step):
        """
        Get the train operation.

        This operation is called within a `tf.control_dependencies(update_ops)`
        block, so implementations do not have to worry about update ops that
        are defined in the calculation of the loss, e.g batch_normalization
        update ops.
        """
        optimizer = tf.train.AdamOptimizer(
            self.params.get('learning_rate', 1e-4))
        return optimizer.minimize(loss, global_step=step)

    @property
    def batch_size(self):
        return self.params.get('batch_size', 64)

    def get_dataset(self, mode, include_labels=True):
        from shapenet.core import get_example_ids, cat_desc_to_id
        from shapenet.core.voxels.config import VoxelConfig
        import random

        cat_desc = self.params.get('cat', 'bed')
        voxel_dim = self.params.get('voxel_dim', 20)

        cat_id = cat_desc_to_id(cat_desc)
        example_ids = list(get_example_ids(cat_id))
        random.seed(0)
        random.shuffle(example_ids)
        n = int(0.8*len(example_ids))
        if mode == 'train':
            example_ids = example_ids[:n]
        else:
            example_ids = example_ids[n:]

        config = VoxelConfig(voxel_dim=voxel_dim)
        dataset = config.get_dataset(cat_id)

        def gen_fn():
            with dataset:
                for example_id in example_ids:
                    yield dataset[example_id].data

        tf_dataset = tf.data.Dataset.from_generator(
            gen_fn, output_types=tf.bool,
            output_shapes=(voxel_dim,)*3)

        def map_fn(voxels):
            voxels = tf.cast(voxels, tf.float32)
            if include_labels:
                return voxels, voxels
            else:
                return voxels

        return tf_dataset.map(map_fn, num_parallel_calls=8)

    def get_train_dataset(self):
        """
        Get a dataset giving features and labels for a single training example.

        Dataset must represent a 2-tuple, each of which can be a tensor, or
        possibly nested list/tuple/dict of tensors.
        """
        return self.get_dataset(tf.estimator.ModeKeys.TRAIN)

    def get_eval_dataset(self):
        """
        Get a dataset giving features and labels for a single eval example.

        Dataset must represent a 2-tuple, each of which can be a tensor, or
        possibly nested list/tuple/dict of tensors.
        """
        return self.get_dataset(tf.estimator.ModeKeys.EVAL)

    def get_predict_dataset(self):
        """
        Get the features for a single prediction example.

        Dataset must represent a tensor, or possibly nested list/tuple/dict of
        tensors.
        """
        return self.get_dataset(
            tf.estimator.ModeKeys.PREDICT, include_labels=False)

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
        threshold = 0.4
        predictions = prediction_data['pred_%.3f' % threshold]
        for pred, gt in zip(predictions, feature_data):
            disp_voxels(pred, color=(0, 1, 0))
            disp_voxels(gt, color=(0, 0, 1))
            show()

    def get_predictions(self, inferences):
        logits = inferences
        probs = tf.nn.sigmoid(logits)
        predictions = {
            'pred_%.3f' % t: tf.greater(probs, t)
            for t in self.thresholds}
        predictions['logits'] = inferences
        predictions['probabilities'] = probs
        return predictions

    @property
    def thresholds(self):
        return np.linspace(0, 1, 11)

    def get_accuracy(self, predictions, labels):
        accuracy = tf.metrics.accuracy(labels, predictions)
        return accuracy

    def get_iou(self, predictions, labels):
        if labels.dtype != tf.bool:
            labels = tf.cast(labels, tf.bool)
        if predictions.dtype != tf.bool:
            predictions = tf.cast(predictions, tf.bool)
        intersection = tf.logical_and(predictions, labels)
        union = tf.logical_or(predictions, labels)
        intersection = tf.reduce_sum(
            tf.cast(intersection, tf.float32), axis=(1, 2, 3))
        union = tf.reduce_sum(
            tf.cast(union, tf.float32), axis=(1, 2, 3))
        iou = intersection / union
        iou = tf.metrics.mean(iou)
        return iou

    def get_eval_metric_ops(self, predictions, labels):
        """Get evaluation metrics. Defaults to empty dictionary."""
        ops = {}
        for threshold in self.thresholds:
            pred = predictions['pred_%.3f' % threshold]
            ops['acc_%.3f' % threshold] = self.get_accuracy(pred, labels)
            ops['iou_%.3f' % threshold] = self.get_iou(pred, labels)
            ops['mean_pred_%.3f' % threshold] = tf.metrics.mean(pred)

        return ops
        # ops = dict(accuracy=self.get_accuracy(predictions, labels),
        #     iou=self.get_iou(predictions, labels))

    def evaluation_report(self, evaluation):
        import matplotlib.pyplot as plt
        print('loss: %.3f' % evaluation['loss'])
        print('global_step: %d' % evaluation['global_step'])
        accs = []
        ious = []
        mps = []
        thresholds = self.thresholds
        print('threshold: acc, iou, mean_pred')
        for threshold in thresholds:
            acc = evaluation['acc_%.3f' % threshold]
            iou = evaluation['iou_%.3f' % threshold]
            mean_pred = evaluation['mean_pred_%.3f' % threshold]
            print('%.3f: %.3f, %.3f, %.3f' % (threshold, acc, iou, mean_pred))
            accs.append(acc)
            ious.append(iou)
            mps.append(mean_pred)

        plt.plot(thresholds, accs, label='accuracy')
        plt.plot(thresholds, ious, label='iou')
        plt.plot(thresholds, mps, label='mean_pred')
        plt.legend(loc=1)
        plt.show()


def register_voxel_ae_family():
    from tf_template.model import register_builder_family
    register_builder_family('voxel_ae', VoxelAEBuilder)


# if __name__ == '__main__':
#     import numpy as np
#     from shapenet.core import get_example_ids, cat_desc_to_id
#     from shapenet.core.voxels.config import VoxelConfig
#
#     mode = 'eval'
#
#     cat_desc = 'bed'
#     voxel_dim = 20
#
#     cat_id = cat_desc_to_id(cat_desc)
#     example_ids = get_example_ids(cat_id)
#     n = int(0.8*len(example_ids))
#     if mode == 'train':
#         example_ids = example_ids[:n]
#     else:
#         example_ids = example_ids[n:]
#
#     config = VoxelConfig(voxel_dim=voxel_dim)
#     dataset = config.get_dataset(cat_id)
#
#     dataset = dataset.map(lambda x: np.sum(x.data))
#
#     sums = []
#     with dataset:
#         for example_id in example_ids:
#             sums.append(dataset[example_id])
#
#     print(np.min(sums), np.max(sums))
