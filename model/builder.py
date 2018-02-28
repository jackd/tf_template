from __future__ import division
import os
import numpy as np
import tensorflow as tf

estimator_dir = os.path.join(os.path.dirname(__file__), 'estimator')


def _tuple_generator(nested_vals):
    iters = tuple(iter(nested_generator(v)) for v in nested_vals)
    try:
        while True:
            yield tuple(next(i) for i in iters)
    except StopIteration:
        pass


def _list_generator(nested_vals):
    iters = tuple(iter(nested_generator(v)) for v in nested_vals)
    try:
        while True:
            yield [next(i) for i in iters]
    except StopIteration:
        pass


def _dict_generator(nested_vals):
    iters = {k: iter(nested_generator(v)) for k, v in nested_vals.items()}
    try:
        while True:
            yield {k: next(i) for k, i in iters.items()}
    except StopIteration:
        pass


def nested_generator(nested_vals):
    if isinstance(nested_vals, np.ndarray):
        return nested_vals
    elif isinstance(nested_vals, (list, tuple)):
        if all(isinstance(v, str) for v in nested_vals):
            return nested_vals
        elif isinstance(nested_vals, tuple):
            return _tuple_generator(nested_vals)
        else:
            return _list_generator(nested_vals)
    elif isinstance(nested_vals, dict):
        return _dict_generator(nested_vals)
    else:
        raise TypeError(
            'Unrecognized type for nested_generator: %s'
            % str(type(nested_vals)))


def initialize_uninitialized_variables(sess):
    global_vars = tf.global_variables()
    is_init = sess.run(
        [tf.is_variable_initialized(var) for var in global_vars])
    init_vars = [v for (v, i) in zip(global_vars, is_init) if not i]
    sess.run(tf.variables_initializer(init_vars))


class ModelBuilder(object):
    """
    Abstract base class for building models.

    Basically an umbrella class containing required functions to build data
    pipelines and `tf.estimator.Estimator`s.

    Concrete implementations must implement:
        * estimator construction:
            * get_inference
            * get_inference_loss
            * get_train_op
        * data pipelines:
            * get_train_dataset
            * get_eval_dataset
            * get_predict_dataset

    Implementations are encouraged to implement:
        * get_predictions
        * get_eval_metrics
        * vis_input_data
        * vis_prediction_data
    """

    def __init__(self, model_id, params):
        self._model_id = model_id
        self._params = params
        self._initial_run = False

    @property
    def initial_run(self):
        return self._intiial_run

    @property
    def needs_custom_initialization(self):
        """Flag indicating whether this model needs custom initialization."""
        return False

    def load_initial_variables(self, graph, sess):
        raise NotImplementedError(
            'Implementation required if `needs_custom_initialization` is True')

    def initialize_variables(self):
        if not self.needs_custom_initialization:
            print('No initialization required: skipping')
            return

        model_dir = self.model_dir
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        elif len(os.listdir(model_dir)) > 0:
            print('Initialization already complete. Skipping.')
            return
        self._intiial_run = True
        try:
            graph = tf.Graph()
            with graph.as_default():
                with tf.Session() as sess:
                    features, labels = self.get_train_inputs()
                    self.get_estimator_spec(features, labels, 'train')
                    self.load_initial_variables(graph, sess)
                    initialize_uninitialized_variables(sess)
                    saver = tf.train.Saver()
                    save_path = os.path.join(self.model_dir, 'model')
                    saver.save(sess, save_path, global_step=0)

        except Exception:
            self._intiial_run = False
            raise
        self._intiial_run = False

    @property
    def model_id(self):
        return self._model_id

    @property
    def params(self):
        return self._params

    @property
    def model_dir(self):
        return os.path.join(estimator_dir, self.model_id)

    @property
    def batch_size(self):
        return self.params['batch_size']

    def get_inference(self, features, mode):
        """Get inferred value of the model."""
        raise NotImplementedError('Abstract method')

    def get_inference_loss(self, inference, labels):
        """Get the loss assocaited with inferences."""
        raise NotImplementedError('Abstract method')

    def get_train_op(self, loss, step):
        """
        Get the train operation.

        This operation is called within a `tf.control_dependencies(update_ops)`
        block, so implementations do not have to worry about update ops that
        are defined in the calculation of the loss, e.g batch_normalization
        update ops.
        """
        raise NotImplementedError('Abstract method')

    def get_train_dataset(self):
        """
        Get a dataset giving features and labels for a single training example.

        Dataset must represent a 2-tuple, each of which can be a tensor, or
        possibly nested list/tuple/dict of tensors.
        """
        raise NotImplementedError('Abstract method')

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
        raise NotImplementedError()

    def vis_prediction_data(
            self, prediction_data, feature_data, label_data=None):
        """
        Function for visualizing a batch of data for training or evaluation.

        All inputs are numpy arrays, or nested dicts/lists of numpy arrays.

        `label_data` may be `None`.

        Not necessary for training/evaluation/infering, but handy for
        debugging.
        """
        raise NotImplementedError()

    def get_predictions(self, inferences):
        """Get predictions. Defaults to the identity, returning inferences."""
        return inferences

    def get_eval_metric_ops(self, predictions, labels):
        """Get evaluation metrics. Defaults to empty dictionary."""
        return dict()

    def get_total_loss(self, inference_loss):
        """
        Get total loss, combining inference loss and regularization losses.

        If no regularization losses, just returns the inference loss.
        """
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(reg_losses) > 0:
            tf.summary.scalar(
                'inference_loss', inference_loss, family='sublosses')
            reg_loss = tf.add_n(reg_losses)
            tf.summary.scalar('reg_loss', reg_loss, family='sublosses')
            loss = inference_loss + reg_loss
        else:
            loss = inference_loss
        return loss

    def get_estimator_spec(self, features, labels, mode, config=None):
        """See `tf.estmator.EstimatorSpec`."""
        inference = self.get_inference(features, mode)
        predictions = self.get_predictions(inference)
        spec_kwargs = dict(mode=mode, predictions=predictions)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(**spec_kwargs)

        inference_loss = self.get_inference_loss(inference, labels)
        loss = self.get_total_loss(inference_loss)
        spec_kwargs['loss'] = loss
        spec_kwargs['eval_metric_ops'] = self.get_eval_metric_ops(
            predictions, labels)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(**spec_kwargs)

        step = tf.train.get_or_create_global_step()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = self.get_train_op(loss=loss, step=step)
        spec_kwargs['train_op'] = train_op

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(**spec_kwargs)

        raise ValueError('Unrecognized mode %s' % mode)

    def get_train_inputs(
            self, shuffle=True, repeat_count=None, shuffle_buffer_size=10000):
        """
        Get all features and labels for training.

        Returns (features, labels), where each of (features, labels) can be
        a tensor, or possibly nested list/tuple/dict.
        """
        dataset = self.get_train_dataset()
        dataset = dataset.shuffle(shuffle_buffer_size).repeat(
            count=repeat_count)
        dataset = dataset.batch(self.batch_size)
        features, labels = dataset.make_one_shot_iterator().get_next()
        return features, labels

    def get_eval_inputs(self):
        """
        Get all features and labels for evlauation.

        Returns (features, labels), where each of (features, labels) can be
        a tensor, or possibly nested list/tuple/dict.
        """
        dataset = self.get_eval_dataset()
        dataset = dataset.batch(self.batch_size)
        features, labels = dataset.make_one_shot_iterator().get_next()
        return features, labels

    def get_predict_inputs(self):
        """
        Abstract method that returns all features required by the model.

        Returned value can be a single tensor, or possibly nested
        list/tuple/dict.
        """
        dataset = self.get_predict_dataset()
        dataset = dataset.batch(self.batch_size)
        features = dataset.make_one_shot_iterator().get_next()
        return features, None

    def get_inputs(self, mode, **kwargs):
        """
        Convenience function for calling inputs with different modes.

        Redirects calls to one of
            * `get_train_inputs`
            * `get_eval_inputs`
            * `get_predict_inputs`.
        """
        if mode == tf.estimator.ModeKeys.TRAIN:
            return self.get_train_inputs(**kwargs)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return self.get_eval_inputs(**kwargs)
        elif mode == tf.esitmator.ModeKeys.INFER:
            return self.get_predict_inputs(**kwargs)

    def get_estimator(self, config=None):
        """Get the `tf.estimator.Estimator` defined by this builder."""
        return tf.estimator.Estimator(
            self.get_estimator_spec, self.model_dir, config=config)

    def train(self, config=None, **train_kwargs):
        """Wrapper around `tf.estimator.Estimator.train`."""
        estimator = self.get_estimator(config=config)
        estimator.train(self.get_train_inputs, **train_kwargs)

    def predict(
            self, mode=tf.estimator.ModeKeys.PREDICT, config=None,
            **predict_kwargs):
        """Wrapper around `tf.estimator.Estimator.predict`."""
        estimator = self.get_estimator(config=config)
        kwargs = dict(shuffle=False, repeat_count=1) if \
            mode == tf.estimator.ModeKeys.TRAIN else {}

        def input_fn():
            return self.get_inputs(mode, **kwargs)

        return estimator.predict(input_fn, **predict_kwargs)

    def eval(
            self, mode=tf.estimator.ModeKeys.EVAL, config=None,
            **eval_kwargs):
        """Wrapper around `tf.estimator.Estimator.eval`."""
        estimator = self.get_estimator(config=config)

        kwargs = dict(shuffle=False, repeat_count=1) if \
            mode == tf.estimator.ModeKeys.TRAIN else {}

        def input_fn():
            return self.get_inputs(mode, **kwargs)
        return estimator.evaluate(self.get_eval_inputs, **eval_kwargs)

    def vis_inputs(self, mode=tf.estimator.ModeKeys.TRAIN):
        """
        Visualize inputs defined by this model according.

        Depends on `vis_example_data` implementation.
        """
        graph = tf.Graph()
        with graph.as_default():
            features, labels = self.get_inputs(mode)

            with tf.train.MonitoredSession() as sess:
                while not sess.should_stop():
                    data = sess.run([features, labels])
                    for record in nested_generator(data):
                        self.vis_example_data(*record)

    def vis_predictions(self, mode=tf.estimator.ModeKeys.PREDICT):
        """
        Visualize inputs and predictions defined by this model.

        Depends on `vis_prediction_data` implementation.
        """
        graph = tf.Graph()
        with graph.as_default():
            features, labels = self.get_inputs(mode)

            predictions = self.get_estimator_spec(
                features, labels, tf.estimator.ModeKeys.PREDICT).predictions
            saver = tf.train.Saver()

            data_tf = [predictions, features]
            if mode != tf.estimator.ModeKeys.PREDICT:
                data_tf.append(labels)

            with tf.train.MonitoredSession() as sess:
                saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))
                while not sess.should_stop():
                    data = sess.run(data_tf)
                    for record in nested_generator(data):
                        self.vis_prediction_data(*record)

    def evaluation_report(self, evaluation):
        print(evaluation)
