from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os


class Evaluator(object):
    def __init__(
            self, data_iter_init, fetches, feed_dict=None, n_eval_steps=None):
        self._data_iter_init = data_iter_init
        self._fetches = fetches
        self._feed_dict = feed_dict
        self._n_eval_steps = n_eval_steps

    def evaluate(self, session):
        session.run(self._data_iter_init)
        i = 0
        while True:
            try:
                values = session.run(self._fetches)
                i += 1
                if self._n_eval_steps is not None and i >= self._n_eval_steps:
                    break
            except tf.errors.OutOfRangeError:
                break
        return values



class EvalWriter(object):
    def __init__(
            self, data_iter_init, eval_metrics, output_dir=None,
            summary_writer=None, n_eval_steps=None):
        self._eval_metrics = eval_metrics
        if summary_writer is None:
            if output_dir is None:
                raise ValueError(
                    'Exactly one of output_dir, summary_writer must be None')
            summary_writer = tf.summary.FileWriter(output_dir)
        else:
            if output_dir is not None:
                raise ValueError(
                    'Exactly one of output_dir, summary_writer must be None')
        self._summary_writer = summary_writer
        self._last_write_step = None
        self._n_eval_steps = n_eval_steps
        self._init = (data_iter_init, self._eval_metrics.init)
        self._fetch = (
            self._eval_metrics.values, self._eval_metrics.summary,
            self._eval_metrics.updates)

    def write(self, session, global_step_value):
        if self._last_write_step == global_step_value:
            tf.logging.info(
                'Summary for step %d already written - skipping'
                % global_step_value)
            return

        session.run(self._init)

        i = 0
        while True:
            try:
                values, summary, _ = session.run(self._fetch)
                i += 1
                if self._n_eval_steps is not None and i >= self._n_eval_steps:
                    break
            except tf.errors.OutOfRangeError:
                break
        self._summary_writer.add_summary(summary, global_step_value)
        self._last_write_step = global_step_value
        tf.logging.info(
            'Finished evaluation at step %d. %s' %
            (global_step_value, str(values)))

    def __call__(self, session, global_step_value):
        return self.write(session, global_step_value)
