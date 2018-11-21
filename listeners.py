from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


class EvalListener(tf.train.CheckpointSaverListener):
    def __init__(
            self, eval_iter_init, eval_metric_ops,
            output_dir=None, summary_writer=None, n_eval_steps=None):
        self._iter_init = eval_iter_init
        self._keys = list(eval_metric_ops.keys())
        eval_metric_ops = [eval_metric_ops[k] for k in eval_metric_ops]
        self._update_ops = [op[1] for op in eval_metric_ops]
        self._values = [op[0] for op in eval_metric_ops]
        self._reset_values_op = tf.variables_initializer(
            tf.local_variables(), name='metrics_reset')
        self._summary = tf.summary.merge([
            tf.summary.scalar(k, v, family='epoch_metrics')
            for k, v in zip(self._keys, self._values)])
        self._summary_writer = None
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

    def _write(self, session, global_step_value):
        if self._last_write_step == global_step_value:
            tf.logging.info(
                'Summary for step %d already written - skipping'
                % global_step_value)
            return

        session.run((self._iter_init, self._reset_values_op))
        fetch = (self._values, self._summary, self._update_ops)

        i = 0
        while True:
            try:
                values, summary, _ = session.run(fetch)
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

    def after_save(self, session, global_step_value):
        return self._write(session, global_step_value)

    def end(self, session, global_step_value):
        self._write(session, global_step_value)
