import tensorflow as tf
import tf_template.model.builder as builder
from tf_template.model.mats.voxel_ae import VoxelAEBuilder


def _get_mobilenet_features(image, mode, load_weights, alpha):
    from mobilenet import MobileNet
    training = mode == tf.estimator.ModeKeys.TRAIN
    tf.keras.backend.set_learning_phase(training)
    weights = 'imagenet' if load_weights else None

    model = MobileNet(
        input_shape=image.shape.as_list()[1:],
        input_tensor=image,
        include_top=False,
        weights=weights)
    return model.output


class ImageEncoderBuilder(builder.ModelBuilder):

    @property
    def needs_custom_initialization(self):
        return True

    @property
    def batch_size(self):
        return self.params.get('batch_size', 2)

    def get_image_features(self, image, mode):
        conv_params = self.params.get('conv_params', {})
        conv_network_id = conv_params.get('model', 'mobilenet')
        if conv_network_id == 'mobilenet':
            image_features = _get_mobilenet_features(
                image, mode, load_weights=self.initial_run,
                alpha=conv_params.get('alpha', 0.25))
        else:
            raise KeyError('Unrecognized conv model: %s' % conv_network_id)
        return image_features

    @property
    def encoder_id(self):
        return self.params.get(
            'encoder_id', 'base_%s' % self.params['cat'])

    def get_encoder_builder(self):
        return VoxelAEBuilder.from_id(self.encoder_id)

    def get_inference(self, features, mode):
        image_features = self.get_image_features(features['image'], mode)
        x = tf.layers.flatten(image_features)
        n_dense = self.params.get('n_dense', [1024])
        embedding_dim = self.get_encoder_builder().embedding_dim
        with tf.variable_scope('image_encoder'):
            for n in n_dense:
                x = tf.layers.dense(x, n, activation=tf.keras.layers.PReLU())
            x = tf.layers.dense(x, embedding_dim)
        return dict(example_id=features['example_id'], encoding=x)

    def get_inference_loss(self, inference, labels):
        return tf.nn.l2_loss(inference['encoding'] - labels)

    def get_train_op(self, loss, step):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.params.get('learning_rate', 1e-8))
        return optimizer.minimize(loss, step)

    def get_cat_id(self):
        from shapenet.core import cat_desc_to_id
        return cat_desc_to_id(self.params['cat'])

    def get_dataset(self, mode, include_labels=True):
        from tf_template.model.mats.ids import get_example_ids
        from shapenet.core.blender_renderings.config import RenderConfig
        from tf_template.model.mats.embedding import get_embeddings_dataset
        from shapenet.image import with_background
        cat_id = self.get_cat_id()
        example_ids = get_example_ids(cat_id, mode)
        shape = self.params.get('image_shape', (192, 256))

        render_config = RenderConfig(shape=shape)
        view_index = self.params.get('view_index', 5)
        image_dataset = render_config.get_dataset(cat_id, view_index)
        encoder_id = self.encoder_id
        embedding_dim = self.get_encoder_builder().embedding_dim

        image_shape = shape + (3,)

        image_dataset.open()

        def get_image_data(example_id):
            image = image_dataset[example_id]
            image = with_background(image, 255)
            return image

        if include_labels:
            embeddings_dataset = get_embeddings_dataset(encoder_id)
            embeddings_dataset.open()

            def map_np_fn(example_id):
                image = get_image_data(example_id)
                embedding = embeddings_dataset[example_id]
                return image, embedding

            def map_tf_fn(example_id):
                image, embedding = tf.py_func(
                    map_np_fn, [example_id], (tf.uint8, tf.float32),
                    stateful=False)
                image.set_shape(image_shape)
                image = tf.image.per_image_standardization(image)
                embedding.set_shape((embedding_dim,))
                return dict(example_id=example_id, image=image), embedding

        else:
            def map_tf_fn(example_id):
                image = tf.py_func(
                    get_image_data, [example_id], tf.uint8, stateful=False)
                image.set_shape(image_shape)
                image = tf.image.per_image_standardization(image)
                return dict(example_id=example_id, image=image)

        return tf.data.Dataset.from_tensor_slices(example_ids).map(map_tf_fn)

    def get_train_dataset(self):
        return self.get_dataset(
            tf.estimator.ModeKeys.TRAIN, include_labels=True)

    def get_eval_dataset(self):
        return self.get_dataset(
            tf.estimator.ModeKeys.EVAL, include_labels=True)

    def get_predict_dataset(self):
        return self.get_dataset(
            tf.estimator.ModeKeys.PREDICT, include_labels=False)

    def vis_example_data(self, feature_data, label_data):
        import numpy as np
        import matplotlib.pyplot as plt
        example_ids = feature_data['example_id']
        images = feature_data['image']
        for example_id, image in zip(example_ids, images):
            image -= np.min(image)
            image /= np.max(image)
            plt.imshow(image)
            plt.title(example_id)
            print(label_data)
            plt.show()

    def vis_prediction_data(self, prediction_data, feature_data, label_data):
        raise NotImplementedError()


def register_image_encoder_family():
    from tf_template.model import register_builder_family
    register_builder_family('image_encoder', ImageEncoderBuilder)
