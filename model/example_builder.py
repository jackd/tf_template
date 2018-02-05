import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from builder import ModelBuilder


def _get_mnist_data(mode):
    data = input_data.read_data_sets(
        os.path.join(
            os.path.realpath(os.path.dirname(__file__)), 'MNIST_data'),
        one_hot=False)
    if mode == tf.estimator.ModeKeys.TRAIN:
        return data.train
    elif mode == tf.estimator.ModeKeys.EVAL:
        return data.validation
    elif mode == tf.estimator.ModeKeys.PREDICT:
        return data.test
    else:
        raise ValueError('Unrecognized mode %s' % mode)


class ExampleModelBuilder(ModelBuilder):
    def get_inference(self, features, mode):
        """Get the logits inferred by the model."""
        training = mode == tf.estimator.ModeKeys.TRAIN
        x = features
        for n in self.params['conv_filters']:
            x = tf.layers.conv2d(
                x, n, 3, padding='VALID', activation=tf.nn.relu)
            x = tf.layers.batch_normalization(
                x, scale=False, training=training)
            x = tf.layers.max_pooling2d(x, 3, 2, padding='VALID')
        x = tf.layers.flatten(x)
        for n in self.params['dense_nodes']:
            x = tf.layers.dense(x, n, activation=tf.nn.relu)
            x = tf.layers.batch_normalization(
                x, scale=False, training=training)
        x = tf.layers.dense(x, self.params['n_classes'])
        return x

    def get_predictions(self, inferences):
        probs = tf.nn.softmax(inferences)
        predictions = tf.argmax(probs, axis=-1)
        return dict(
            logits=inferences,
            probabilities=probs,
            predictions=predictions
        )

    def get_accuracy(self, predictions, labels):
        accuracy = tf.metrics.accuracy(labels, predictions['predictions'])
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def get_eval_metric_ops(self, predictions, labels):
        return dict(accuracy=self.get_accuracy(predictions, labels))

    def get_inference_loss(self, inference, labels):
        return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=inference))

    def get_train_op(self, loss, step):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.params['learning_rate'])
        return optimizer.minimize(loss, global_step=step)

    def get_train_dataset(self):
        data = _get_mnist_data(tf.estimator.ModeKeys.TRAIN)
        images = tf.constant(data.images.reshape(-1, 28, 28, 1))
        labels = tf.constant(data.labels, dtype=tf.int32)
        tensors = images, labels
        dataset = tf.data.Dataset.from_tensor_slices(tensors)
        return dataset

    def get_eval_dataset(self):
        data = _get_mnist_data(tf.estimator.ModeKeys.EVAL)
        images = tf.constant(data.images.reshape(-1, 28, 28, 1))
        labels = tf.constant(data.labels, dtype=tf.int32)
        tensors = images, labels
        dataset = tf.data.Dataset.from_tensor_slices(tensors)
        return dataset

    def get_predict_dataset(self):
        data = _get_mnist_data(tf.estimator.ModeKeys.INFER)
        images = tf.constant(data.images.reshape(-1, 28, 28, 1))
        dataset = tf.data.Dataset.from_tensor_slices(images)
        return dataset

    def vis_example_data(self, feature_data, label_data):
        import matplotlib.pyplot as plt
        for image, label in zip(feature_data, label_data):
            plt.imshow(image[..., 0], cmap='gray')
            plt.title(str(label))
            plt.show()

    def vis_prediction_data(self, prediction_data, feature_data, label_data):
        # import inside the function to prevent loading when not visualizing.
        import matplotlib.pyplot as plt
        images = feature_data[..., 0]
        preds = prediction_data['predictions']
        all_probs = prediction_data['probabilities']

        for image, pred, probs, label in zip(
                images, preds, all_probs, label_data):
            plt.imshow(image, cmap='gray')
            prob = probs[pred]
            pred_str = 'prediction = %d, prob = %.5f%%' % (pred, 100*prob)
            if label_data is None:
                plt.title(pred_str)
            else:
                plt.title('label = %d, %s' % (label, pred_str))
            plt.show()


def register_example_family():
    from tf_template.model import register_builder_family
    register_builder_family('example', ExampleModelBuilder)
