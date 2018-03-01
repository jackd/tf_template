from __future__ import division
import numpy as np
import tensorflow as tf
import tf_template.model.builder as builder
from image_encoder import ImageEncoderBuilder
from voxel_ae import VoxelAEBuilder
from tf_template.model.mats.ids import get_example_ids


class FineTuneModelBuilder(builder.ModelBuilder):
    def get_cat_id(self):
        from shapenet.core import cat_desc_to_id
        return cat_desc_to_id(self.params['cat'])

    @property
    def voxel_ae_id(self):
        return self.params.get('voxel_ae_id', 'base_%s' % self.params['cat'])

    @property
    def image_encoder_id(self):
        return self.params.get(
            'image_encoder_id', 'base_image_%s' % self.params['cat'])

    @property
    def image_encoder_builder(self):
        return ImageEncoderBuilder.from_id(self.image_encoder_id)

    @property
    def voxel_ae_builder(self):
        return VoxelAEBuilder.from_id(self.voxel_ae_id)

    @property
    def needs_custom_initialization(self):
        """Flag indicating whether this model needs custom initialization."""
        return True

    def load_initial_variables(self, graph, sess):
        self.voxel_ae_builder.load_decoder_variables(graph, sess)
        self.image_encoder_builder.load_encoder_variables(graph, sess)

    def get_inference(self, features, mode):
        """Get inferred value of the model."""
        example_id, image = (features[k] for k in ('example_id', 'image'))
        encoding = self.image_encoder_builder.encode(image, mode)
        logits = self.voxel_ae_builder.decode(encoding, mode)
        return dict(
            example_id=example_id, encoding=encoding, logits=logits)

    def get_inference_loss(self, inference, labels):
        """Get the loss assocaited with inferences."""
        logits = inference['logits']
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels))

    def get_train_op(self, loss, step):
        """
        Get the train operation.

        This operation is called within a `tf.control_dependencies(update_ops)`
        block, so implementations do not have to worry about update ops that
        are defined in the calculation of the loss, e.g batch_normalization
        update ops.
        """
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.params.get('learning_rate', 1e-8))
        return optimizer.minimize(loss, step)

    def get_dataset(self, mode, include_labels=True):
        cat_id = self.get_cat_id()
        example_ids = get_example_ids(cat_id, mode)

        image_builder = self.image_encoder_builder
        image_dataset = image_builder.get_image_dataset()
        image_dataset.open()

        if include_labels:
            voxel_builder = self.voxel_ae_builder
            voxel_dataset = voxel_builder.get_voxel_dataset()
            voxel_dataset.open()

            def gen_fn():
                for example_id in example_ids:
                    image = image_dataset[example_id]
                    voxels = voxel_dataset[example_id]
                    yield example_id, image, voxels

            def map_fn(example_id, image, voxels):
                image = image_builder.preprocess_image(image)
                voxels = voxel_builder.preprocess_voxels(voxels)
                features = dict(image=image, example_id=example_id)
                return features, voxels

            dataset = tf.data.Dataset.from_generator(
                gen_fn, output_types=(tf.string, tf.uint8, tf.bool),
                output_shapes=(
                    (),
                    image_builder.input_shape + (3,),
                    (voxel_builder.voxel_dim,)*3))
        else:
            def gen_fn():
                for example_id in example_ids:
                    image = image_dataset[example_id]
                    image = image_builder.preprocess_image(image)
                    yield example_id, image

            def map_fn(example_id, image, voxels):
                image = tf.image.per_image_standardization(image)
                voxels = tf.cast(voxels, tf.float32)
                features = dict(image=image, example_id=example_id)
                return features

            dataset = tf.data.Dataset.from_generator(
                gen_fn, output_types=(tf.string, tf.uint8, tf.bool),
                output_shapes=(
                    (),
                    image_builder.input_shape + (3,)))

        dataset = dataset.map(map_fn, num_parallel_calls=8)
        return dataset

    @property
    def batch_size(self):
        return self.params.get('batch_size', 2)

    def get_train_dataset(self):
        return self.get_dataset(tf.estimator.ModeKeys.TRAIN)

    def get_eval_dataset(self):
        return self.get_dataset(tf.estimator.ModeKeys.EVAL)

    def get_predict_dataset(self):
        return self.get_dataset(
            tf.estimator.ModeKeys.PREDICT, include_labels=False)

    def vis_example_data(self, feature_data, label_data):
        import matplotlib.pyplot as plt
        from shapenet.mayavi_vis import vis_voxels
        from mayavi import mlab
        image = feature_data['image']
        image -= np.min(image)
        image /= np.max(image)
        plt.imshow(image)
        vis_voxels(label_data, color=(0, 0, 1))
        plt.show(block=False)
        mlab.show()
        plt.close()

    def vis_prediction_data(
            self, prediction_data, feature_data, label_data=None):
        import matplotlib.pyplot as plt
        from shapenet.mayavi_vis import vis_voxels
        from mayavi import mlab
        image = feature_data['image']
        threshold = 0.4
        image -= np.min(image)
        image /= np.max(image)
        plt.imshow(image)
        mlab.figure()
        pred = prediction_data['pred_%.3f' % threshold]
        vis_voxels(pred)
        if label_data is not None:
            mlab.figure()
            vis_voxels(label_data, color=(0, 0, 1))
            intersection = np.sum(np.logical_and(pred, label_data))
            union = np.sum(np.logical_or(pred, label_data))
            iou = intersection / union
            print('iou: %.5f' % iou)
        plt.show(block=False)
        mlab.show()
        plt.close()


def register_fine_tune_family():
    from tf_template.model import register_builder_family
    register_builder_family('fine_tune', FineTuneModelBuilder)
