from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
# from .eval_writer import EvalWriter

class InterruptorListener(object):
    def begin(self):
        pass

    def before_interrupt(self, session, global_step_value):
        pass

    def after_interrupt(self, session, global_step_value):
        pass


# class InterruptorListenerBase(InterruptorListener):
#     def __init__(self, begin_fn=None, before_save_fn=None, after_save_fn=None):
#         self._begin_fn = begin_fn or lambda *args: None
#         self._before_save_fn = before_interrupt_fn or lambda *args: None
#         self._after_save_fn = after_interrupt_fn or lambda *args: None

#     def begin(self):
#         return self._begin()

#     def before_interrupt(self, session, global_step_value):
#         return self._before_interrupt_fn(session, global_step_value)

#     def after_interrupt(self, session, global_step_value):
#         return self._after_interrupt_fn(session, global_step_value)


class InterruptorHook(tf.train.SessionRunHook):
    def __init__(
            self, callback, every_steps=None, every_secs=None, listeners=()):
        self._callback = callback
        self._timer = tf.train.SecondOrStepTimer(
            every_secs=every_secs, every_steps=every_steps)
        self._listeners = tuple(listeners)

    def after_create_session(self, session, coord):
        self._sess = session

    def begin(self):
        self._timer.reset()
        self._iter_count = 0
        self._step = tf.train.get_global_step()
        for listener in self._listeners:
            listener.begin()

    def before_run(self, run_context):
        self._should_trigger = self._timer.should_trigger_for_step(
            self._iter_count)
        if self._should_trigger:
            return tf.train.SessionRunArgs(dict(step=self._step))
        else:
            return None

    def after_run(self, run_context, run_values):
        if self._should_trigger:
            sess = self._sess
            step = run_values.results['step']
            for listener in self._listeners:
                listener.before_interrupt(sess, step)
            self._callback(self._sess, step)
            for listener in self._listeners:
                listener.after_interrupt(self._sess, step)
            self._timer.update_last_triggered_step(self._iter_count)
        self._iter_count += 1

class AtEndHook(tf.train.SessionRunHook):
    def __init__(self, callback):
        self._callback = callback

    def end(self, session):
        self._callback(session)


class PeriodicEvalHook(tf.train.SessionRunHook):
    def __init__(
            self, writer, every_steps=None, every_secs=None, at_end=True):
        """
        Args:
            writer: an EvalWriter instance.
        """
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


class ResetSummaryHook(tf.train.SessionRunHook):
    def __init__(
            self, summary_writer, summary_op, reset_op, every_steps=None,
            every_secs=None):
        self._writer = summary_writer
        self._timer = tf.train.SecondOrStepTimer(
            every_secs=every_secs, every_steps=every_steps)
        self._reset_op = reset_op
        self._summary_op = summary_op

    def after_create_session(self, session, coord):
        self._sess = session
        session.run(self._reset_op)

    def begin(self):
        self._timer.reset()
        self._iter_count = 0
        self._step = tf.train.get_global_step()

    def before_run(self, run_context):
        self._should_trigger = self._timer.should_trigger_for_step(
            self._iter_count)
        if self._should_trigger:
            return tf.train.SessionRunArgs(
                dict(step=self._step, summary=self._summary_op))
        else:
            return None

    def after_run(self, run_context, run_values):
        if self._should_trigger:
            if self._iter_count != 0:
                results = run_values.results
                self._writer.add_summary(
                    results['summary'], results['step'])
                self._sess.run(self._reset_op)
            self._timer.update_last_triggered_step(self._iter_count)
        self._iter_count += 1




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
