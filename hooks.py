from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
# from .eval_writer import EvalWriter

class AtEndHook(tf.train.SessionRunHook):
    def __init__(self, callback):
        self._callback = callback

    def end(self, session):
        self._callback(session)


class PeriodicEvalHook(tf.train.SessionRunHook):
    def __init__(
            self, writer, every_steps=None, every_secs=None, at_end=True):
        if every_steps is None and every_secs is None:
            every_secs = 600
        self._writer = writer
        self._timer = tf.train.SecondOrStepTimer(
            every_secs=every_secs, every_steps=every_steps)
        self._at_end = at_end

    def after_create_session(self, session, coord):
        self._sess = session

    def begin(self):
        self._timer.reset()
        self._iter_count = 0
        self._step = tf.train.get_global_step()

    def before_run(self, run_context):
        self._should_trigger = self._timer.should_trigger_for_step(
            self._iter_count)
        if self._should_trigger:
            return tf.train.SessionRunArgs(dict(step=self._step))
        else:
            return None

    def after_run(self, run_context, run_values):
        if self._should_trigger:
            self._writer.write(self._sess, run_values.results['step'])
            self._timer.update_last_triggered_step(self._iter_count)
        self._iter_count += 1

    def end(self, session):
        if self._at_end:
            s = session.run(self._step)
            self._writer.write(session, s)





# class MinimalCheckpointSaverHook(tf.train.SessionRunHook):
#     def __init__(
#             self, checkpoint_dir, saver, save_secs=None, save_steps=None,
#             checkpoint_basename="model.ckpt",):
#         self._checkpoint_dir = checkpoint_dir
#         self._saver = saver
#         self._timer = tf.train.SecondOrStepTimer(
#             every_secs=save_secs,
#             every_steps=save_steps)
#         self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)

#     def begin(self):
#         self._global_step_tensor = tf.train.get_or_create_global_step()

#     def after_create_session(self, session, coord):
#         global_step = session.run(self._global_step_tensor)
#         self._save(session, global_step)
#         self._timer.update_last_triggered_step(global_step)

#     def _save(self, session, step):
#         self._saver.save(session, self._save_path, global_step=step)
