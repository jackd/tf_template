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
    @staticmethod
    def from_id(model_id):
        from tf_template.model import load_params
        params = load_params(model_id)
        if params['family'] != 'image_encoder':
            raise ValueError(
                'parameters for model %s have family other than '
                '"image_encoder"')
        return ImageEncoderBuilder(model_id, params)

    @property
    def needs_custom_initialization(self):
        return True

    def load_initial_variables(self, graph, sess):
        pass

    _encoder_scope = 'image_encoder'

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

    def encode(self, image, mode):
        n_dense = self.params.get('n_dense', [1024])
        embedding_dim = self.get_encoder_builder().embedding_dim
        with tf.variable_scope(ImageEncoderBuilder._encoder_scope):
            image_features = self.get_image_features(image, mode)
            x = tf.layers.flatten(image_features)
            for n in n_dense:
                x = tf.layers.dense(x, n, activation=tf.keras.layers.PReLU())
            x = tf.layers.dense(x, embedding_dim)
        return x

    def get_inference(self, features, mode):
        encoding = self.encode(features['image'], mode)
        return dict(example_id=features['example_id'], encoding=encoding)

    def get_inference_loss(self, inference, labels):
        return tf.nn.l2_loss(inference['encoding'] - labels)

    def get_train_op(self, loss, step):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.params.get('learning_rate', 1e-8))
        return optimizer.minimize(loss, step)

    def get_cat_id(self):
        from shapenet.core import cat_desc_to_id
        return cat_desc_to_id(self.params['cat'])

    @property
    def input_shape(self):
        return self.params.get('image_shape', (192, 256))

    def get_image_dataset(self):
        from shapenet.image import with_background
        from shapenet.core.blender_renderings.config import RenderConfig
        render_config = RenderConfig(shape=self.input_shape)
        view_index = self.params.get('view_index', 5)
        cat_id = self.get_cat_id()
        image_dataset = render_config.get_dataset(cat_id, view_index)
        return image_dataset.map(lambda x: with_background(x, 255))

    def preprocess_image(self, image):
        return tf.image.per_image_standardization(image)

    def get_dataset(self, mode, include_labels=True):
        from tf_template.model.mats.ids import get_example_ids
        from tf_template.model.mats.embedding import get_embeddings_dataset
        cat_id = self.get_cat_id()
        example_ids = get_example_ids(cat_id, mode)
        shape = self.input_shape

        encoder_id = self.encoder_id
        embedding_dim = self.get_encoder_builder().embedding_dim

        image_shape = shape + (3,)

        image_dataset = self.get_image_dataset()
        image_dataset.open()

        if include_labels:
            embeddings_dataset = get_embeddings_dataset(encoder_id)
            embeddings_dataset.open()

            def map_np_fn(example_id):
                image = image_dataset[example_id]
                embedding = embeddings_dataset[example_id]
                return image, embedding

            def map_tf_fn(example_id):
                image, embedding = tf.py_func(
                    map_np_fn, [example_id], (tf.uint8, tf.float32),
                    stateful=False)
                image.set_shape(image_shape)
                image = self.preprocess_image(image)
                embedding.set_shape((embedding_dim,))
                return dict(example_id=example_id, image=image), embedding

        else:
            def map_tf_fn(example_id):
                image = tf.py_func(
                    lambda x: image_dataset[x], [example_id], tf.uint8,
                    stateful=False)
                image.set_shape(image_shape)
                image = self.preprocess_image(image)
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
        example_id = feature_data['example_id']
        image = feature_data['image']
        image -= np.min(image)
        image /= np.max(image)
        plt.imshow(image)
        plt.title(example_id)
        print(label_data)
        plt.show()

    def vis_prediction_data(
            self, prediction_data, feature_data, label_data=None):
        raise NotImplementedError()

    def load_encoder_variables(self, graph, sess):
        var_list = tf.get_collection(
            tf.GraphKeys.VARIABLES, scope=ImageEncoderBuilder._encoder_scope)
        saver = tf.train.Saver(var_list)
        saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))


def register_image_encoder_family():
    from tf_template.model import register_builder_family
    register_builder_family('image_encoder', ImageEncoderBuilder)
