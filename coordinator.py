from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
ModeKeys = tf.estimator.ModeKeys


class Coordinator(object):
    def __init__(
            self, data_source, inference_model, train_model, model_dir,
            eval_metric_ops_fn=None, custom_hooks_fn=None):
        self._inference_model = inference_model
        self._data_source = data_source
        self._train_model = train_model
        self._model_dir = model_dir
        self._eval_metric_ops_fn = eval_metric_ops_fn
        self._custom_hooks_fn = custom_hooks_fn

    @property
    def inference_model(self):
        return self._inference_model

    @property
    def train_model(self):
        return self._train_model

    @property
    def data_source(self):
        return self._data_source

    @property
    def model_dir(self):
        return self._model_dir

    def get_eval_metric_ops(self, predictions, labels):
        if self._eval_metric_ops_fn is None:
            return None
        else:
            return self._eval_metric_ops_fn(predictions, labels)

    def get_custom_hooks(self, mode):
        if self._custom_hooks_fn is None:
            return None
        else:
            return self._custom_hooks_fn(self, mode)

    def get_estimator_spec(self, features, labels, mode):
        inference = self.inference_model.get_inference(features, mode)
        predictions = self.inference_model.get_predictions(
            features, inference)
        kwargs = dict(
            mode=mode,
            predictions=predictions
        )
        if mode == ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(**kwargs)

        loss = self.train_model.get_total_loss(inference, labels)
        kwargs['eval_metric_ops'] = self.get_eval_metric_ops(
            predictions, labels)

        kwargs['loss'] = loss
        if mode == ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(**kwargs)
        kwargs['train_op'] = self.train_model.get_train_op(loss)
        return tf.estimator.EstimatorSpec(**kwargs)

    def get_inputs(self, mode, **input_kwargs):
        if 'batch_size' not in input_kwargs:
            input_kwargs['batch_size'] = self.train_model.batch_size
        return self.data_source.get_inputs(
            mode, **input_kwargs)

    def get_estimator(self, config=None):
        if tf.train.latest_checkpoint(self.model_dir) is None:
            warm_start = self.inference_model.get_warm_start_settings()
        else:
            warm_start = None
        return tf.estimator.Estimator(
            self.get_estimator_spec, model_dir=self.model_dir, config=config,
            warm_start_from=warm_start)

    def train(self, config=None):
        estimator = self.get_estimator(config=config)

        return estimator.train(
            lambda: self.get_inputs(ModeKeys.TRAIN),
            max_steps=self.train_model.max_steps,
            hooks=self.get_custom_hooks(ModeKeys.TRAIN))

    def evaluate(self, config=None, input_kwargs={}, **eval_kwargs):
        estimator = self.get_estimator(config=config)
        input_kwargs.setdefault('mode', ModeKeys.EVAL)
        return estimator.evaluate(
            lambda: self.get_inputs(**input_kwargs),
            hooks=self.get_custom_hooks(ModeKeys.EVAL),
            **eval_kwargs)

    def predict(self, config=None, input_kwargs={}, **predict_kwargs):
        estimator = self.get_estimator(config=config)
        input_kwargs = input_kwargs.copy()
        input_kwargs.setdefault('mode', ModeKeys.PREDICT)
        return estimator.predict(
            lambda: self.get_inputs(**input_kwargs),
            hooks=self.get_custom_hooks(ModeKeys.PREDICT),
            **predict_kwargs)

    def train_and_evaluate(self, config=None, **eval_spec_kwargs):
        estimator = self.get_estimator(config=config)
        train_spec = tf.estimator.TrainSpec(
            lambda: self.get_inputs(ModeKeys.TRAIN),
            max_steps=self.train_model.max_steps,
            hooks=self.get_custom_hooks(ModeKeys.TRAIN))
        eval_spec = tf.estimator.EvalSpec(
            lambda: self.get_inputs(ModeKeys.EVAL),
            hooks=self.get_custom_hooks(ModeKeys.EVAL),
            **eval_spec_kwargs)
        return tf.estimator.train_and_evaluate(
            estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)

    def vis_predictions(
            self, config=None, data_mode=ModeKeys.PREDICT, **predict_kwargs):
        nest = tf.contrib.framework.nest
        if data_mode is None:
            data_mode = ModeKeys.PREDICT

        # def get_predictions_spec(features, labels, mode):
        #     base_spec = self.get_estimator_spec(features, labels, mode)
        #     predictions = dict(
        #         predictions=base_spec.predictions,
        #         features=features)
        #     if labels is not None:
        #         predictions['labels'] = labels
        #     return tf.estimator.EstimatorSpec(
        #         predictions=predictions, mode=mode)
        #
        # estimator = tf.estimator.Estimator(
        #     model_fn=get_predictions_spec, model_dir=self.model_dir,
        #     config=config)
        # for prediction in estimator.predict(
        #         lambda: self.get_inputs(mode=data_mode), **predict_kwargs):
        #     self.vis_prediction_data(**prediction)

        graph = tf.Graph()
        with graph.as_default():
            features, labels = self.get_inputs(mode=data_mode)
            spec = self.get_estimator_spec(
                features, labels, mode=ModeKeys.PREDICT)
            predictions = spec.predictions

            session_creator = tf.train.ChiefSessionCreator(
                config=config.session_config)
            tensors = dict(features=features, predictions=predictions)
            if labels is not None:
                tensors['labels'] = labels
            saver = tf.train.Saver()

            with tf.train.MonitoredSession(
                    session_creator=session_creator) as sess:
                saver.restore(
                    sess, tf.train.latest_checkpoint(self.model_dir))
                while not sess.should_stop():
                    data = sess.run(tensors)
                    flat_data = nest.flatten(data)
                    for record in zip(*flat_data):
                        record = nest.pack_sequence_as(tensors, record)
                        self.vis_prediction_data(**record)

    def prediction_vis(self, features, predictions, labels=None):
        vis = []
        vis.append(self.data_source.input_vis(features, labels))
        vis.append(self.inference_model.prediction_vis(predictions))
        return vis

    def vis_prediction_data(
            self, features, predictions, labels=None):
        from .util import maybe_stop
        from .visualization import get_vis
        vis = self.prediction_vis(features, predictions, labels)
        vis = get_vis(*vis)
        vis.show(block=False)
        maybe_stop()
        vis.close()

    def create_profile(
            self, data_mode=ModeKeys.TRAIN, inference_mode=ModeKeys.TRAIN,
            batch_size=None, path=None, skip_runs=10, use_dummy_inputs=False,
            config=None):
        import os
        from tf_toolbox.profile import create_profile
        if batch_size is None:
            batch_size = self.train_model.batch_size

        def graph_fn():
            if use_dummy_inputs:
                features, labels = self.data_source.get_dummy_inputs(
                    data_mode, batch_size)
            else:
                features, labels = self.data_source.get_inputs(
                    data_mode, batch_size)
            spec = self.get_estimator_spec(features, labels, inference_mode)
            if inference_mode == tf.estimator.ModeKeys.PREDICT:
                return spec.predictions
            elif inference_mode == tf.estimator.ModeKeys.EVAL:
                return spec.eval_metric_ops
            elif inference_mode == tf.estimator.ModeKeys.TRAIN:
                return spec.train_op
            else:
                raise ValueError(
                    'Invalid inference_mode "%s"' % inference_mode)

        if path is None:
            fn = 'profile_dummy.json' if use_dummy_inputs else 'profile.json'
            path = os.path.join(self.model_dir, fn)

        create_profile(graph_fn, path, skip_runs=skip_runs, config=config)

    def report_train_tests(
            self, variable_change_test=True, update_ops_test=True,
            steps=5, config=None):
        import tf_toolbox.testing

        def get_train_op():
            features, labels = self.get_inputs(ModeKeys.TRAIN)
            return self.get_estimator_spec(
                features, labels, ModeKeys.TRAIN).train_op

        if variable_change_test:
            tf_toolbox.testing.report_train_val_changes(
                get_train_op, steps=steps, config=config)

        if update_ops_test:
            tf_toolbox.testing.report_update_ops_run(
                get_train_op, config=config)

    def _clean(self):
        import shutil
        if os.path.isdir(self.model_dir):
            shutil.rmtree(self.model_dir)
            print('Removed %s' % self.model_dir)
        else:
            print('No model dir at %s' % self.model_dir)

    def clean(self, confirm=True):
        from .util import get_input
        if confirm:
            inp = get_input(
                'Definitely remove saved data in %s? (y/N)' % self.model_dir)
            inp = inp.lower()
            if inp == 'y':
                pass
            elif inp == 'n':
                print('NOT removing...')
                return
            else:
                raise ValueError('Invalid input "%s"' % inp)
        self._clean()

    def count_trainable_parameters(self, mode=ModeKeys.TRAIN):
        graph = tf.Graph()
        with graph.as_default():
            features, labels = self.get_inputs(mode, batch_size=1)
            self.get_estimator_spec(features, labels, mode)
            return sum(
                t.shape.num_elements() for t in tf.trainable_variables())
