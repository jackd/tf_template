"""Experimental features - unlikely to be stable."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import six
import collections
import tensorflow as tf


MetricOps = collections.namedtuple(
    'MetricOps', ['summary', 'updates', 'values', 'init'])


def split_metric_ops(metric_ops, name, family=None):
    if isinstance(name, six.string_types) or name is None:
        local_vars = tf.local_variables(scope=name)
    elif hasattr(name, '__iter__'):
        local_vars = []
        for n in name:
            local_vars.extend(tf.local_variables(scope=n))
        local_vars = tuple(set(local_vars))
    else:
        raise TypeError(
            'name should have __iter__ or be string or None, got %s'
            % str(name))
    init = tuple(v.initializer for v in local_vars)
    summary = tf.summary.merge(tuple(
        tf.summary.scalar(k, v[0], family=family)
        for k, v in metric_ops.items()))
    updates = tuple(v[1] for v in metric_ops.items())
    values = {k: v[0] for k, v in metric_ops.items()}
    return MetricOps(
        summary=summary, updates=updates, values=values, init=init)


def only_one(a, b, default_a, a_key, b_key):
    if a is None:
        if b is None:
            return default_a, None
        else:
            return None, b
    else:
        if b is None:
            return a, None
        else:
            raise ValueError(
                'Cannot provide both values: %s, %s' % (a_key, b_key))



def custom_train_and_evaluate(
        coord,
        save_checkpoints_secs=None, save_checkpoints_steps=None,
        save_summary_steps=None,
        save_summary_secs=None,
        log_step_count_steps=100,
        eval_every_secs=None, eval_every_steps=None,
        n_eval_steps=None):
    import tensorflow as tf
    from .hooks import InterruptorHook
    from .hooks import ResetSummaryHook
    # from .hooks import InterruptorListener
    # from .eval_writer import EvalWriter
    # from .hooks import PeriodicEvalHook
    # from .listeners import EvalListener
    # from .hooks import AtEndHook
    ModeKeys = tf.estimator.ModeKeys
    save_checkpoints_secs, save_checkpoints_steps = only_one(
        save_checkpoints_secs, save_checkpoints_steps, 600,
        'save_checkpoint_secs', 'save_checkpoint_steps')
    save_summary_steps, save_summary_secs = only_one(
        save_summary_steps, save_summary_secs, 100,
        'save_summary_steps', 'save_summary_secs')
    max_steps = coord.train_model.max_steps

    model_dir = coord.model_dir
    eval_dir = os.path.join(model_dir, 'eval')
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    graph = tf.Graph()
    with graph.as_default():
        ds = coord.data_source
        batch_size = coord.train_model.batch_size
        train_ds = ds.get_inputs(mode=ModeKeys.TRAIN, batch_size=batch_size)
        eval_ds = ds.get_inputs(
            mode=ModeKeys.EVAL, batch_size=batch_size, repeat_count=1)
        if not all(
                isinstance(d, tf.data.Dataset) for d in (train_ds, eval_ds)):
            raise RuntimeError('get_inputs must return a tf.data.Dataset')

        train_iter = train_ds.make_one_shot_iterator()
        eval_iter = eval_ds.make_initializable_iterator()

        train_handle = train_iter.string_handle()
        eval_handle = eval_iter.string_handle()

        mode = tf.placeholder_with_default('eval', shape=(), name='mode')
        handle = tf.cond(
            tf.equal(mode, ModeKeys.TRAIN),
            lambda: train_handle, lambda: eval_handle)

        iterator = tf.data.Iterator.from_string_handle(
            handle, train_ds.output_types, train_ds.output_shapes)

        features, labels = iterator.get_next()
        spec = coord.get_estimator_spec(features, labels, mode)
        eval_metric_ops = spec.eval_metric_ops
        train_op = spec.train_op
        loss = spec.loss
        predictions = spec.predictions

        step = tf.train.get_global_step()

        with tf.variable_scope('train_metrics'):
            train_metric_ops = coord.get_eval_metric_ops(predictions, labels)
            train_metric_ops['loss'] = tf.metrics.mean(loss)
            train_metrics = split_metric_ops(train_metric_ops, 'train_metrics')

        train_summary = tf.summary.merge_all()

        if 'loss' in eval_metric_ops:
            raise RuntimeError(
                'Reserved key already present in eval_metric_ops: "loss"')
        with tf.variable_scope('loss_metrics'):
            eval_metric_ops['loss'] = tf.metrics.mean(loss)

        eval_metrics = split_metric_ops(
            eval_metric_ops, name=('eval_metrics', 'loss_metrics'),
            family='eval_metrics')

        saver = tf.train.Saver()

        # eval_listener = EvalListener(
        # eval_writer = EvalWriter(
        #     eval_iter.initializer, eval_metrics,
        #     summary_writer=tf.summary.FileWriter(logdir=eval_dir),
        #     n_eval_steps=n_eval_steps)
        eval_writer = tf.summary.FileWriter(eval_dir)

        def interrupt_body(sess, step):
            sess.run((eval_iter.initializer, eval_metrics.init))
            i = 0
            while True:
                try:
                    summary, values, _ = sess.run((
                        eval_metrics.summary,
                        eval_metrics.values,
                        eval_metrics.updates,
                    ))
                    i += 1
                    if n_eval_steps is not None and i >= n_eval_steps:
                        break
                except tf.errors.OutOfRangeError:
                    break
            eval_writer.add_summary(summary, step)
            tf.logging.info('Step %d evaluation: %s' % (step, str(values)))

        eval_hook = InterruptorHook(
            interrupt_body, every_secs=eval_every_secs,
            every_steps=eval_every_steps)


        # eval_hook = PeriodicEvalHook(
        #     eval_writer,
        #     every_steps=eval_every_steps, every_secs=eval_every_secs,
        #     at_end=True)

        checkpoint_hook = tf.train.CheckpointSaverHook(
                model_dir, save_secs=save_checkpoints_secs,
                save_steps=save_checkpoints_steps,
                saver=saver)
        logging_hook = tf.train.LoggingTensorHook(
            dict(loss=loss, step=step), every_n_iter=log_step_count_steps,
            formatter=lambda x:
                'loss at step %d: %s' % (x['step'], str(x['loss'])))

        hooks = [
            checkpoint_hook,
            logging_hook,
            tf.train.NanTensorHook(loss),
            eval_hook,
        ]

        if train_summary is not None:
            train_writer = tf.summary.FileWriter(logdir=model_dir)
            # summary_hook = tf.train.SummarySaverHook(
            #     save_steps=save_summary_steps,
            #     summary_writer=train_writer,
            #     summary_op=train_summary)
            summary_hook = ResetSummaryHook(
                train_writer, train_summary, train_metrics.init,
                every_steps=save_summary_steps, every_secs=save_summary_secs)
            hooks.append(summary_hook)

        session_creator = tf.train.ChiefSessionCreator(
            checkpoint_dir=model_dir)

        s = None

        with tf.train.MonitoredSession(
                session_creator=session_creator, hooks=hooks) as sess:
            train_feed = {mode: ModeKeys.TRAIN}
            i = 0
            while True:
                s, _, __ = sess.run(
                    (step, train_op, train_metrics.updates), train_feed)
                i += 1
                if s >= max_steps:
                    break
