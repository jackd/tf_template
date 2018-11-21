"""Experimental features - unlikely to be stable."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os


def register_experimental():
    from .cli import register_coord_fn, FLAGS
    def f2(coord):
        kwargs = dict(
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            save_checkpoints_secs=FLAGS.save_checkpoints_secs,
            save_summary_steps=FLAGS.save_summary_steps,
            log_step_count_steps=FLAGS.log_step_count_steps,
            n_eval_steps=FLAGS.n_eval_steps,
        )
        return custom_train_and_evaluate2(coord, **kwargs)

    for k in ('custom_train_and_evaluate', 'custom_train_and_eval'):
        register_coord_fn(k, custom_train_end_evaluate)
        register_coord_fn('%s2' % k, f2)



def _custom_train_and_evaluate(coord):
    from absl import flags
    FLAGS = flags.FLAGS
    return custom_train_end_evaluate(
        coord, save_checkpoints_secs=FLAGS.save_checkpoints_secs or 600,
        save_summary_steps=FLAGS.save_summary_steps,
        log_step_count_steps=FLAGS.log_step_count_steps)


def custom_train_end_evaluate(
        coord, epochs_per_eval=1, save_checkpoints_secs=600,
        save_summary_steps=100, log_step_count_steps=100):
    """
    Custon train and evaluate loop based on reinitializable iterators.

    Requires coord.data_source.get_inputs to return a dataset, and take a
    `repeats` kwargs.

    `mode` passed to `InferenceModel` will also be a tensor, rather than a
    string. training = mode == 'train' will no longer work as intended - use
    training = tf.equal(mode, 'train')
    """
    import tensorflow as tf
    ModeKeys = tf.estimator.ModeKeys
    model_dir = coord.model_dir
    eval_summary_dir = os.path.join(model_dir, 'eval')
    train_summary_dir = os.path.join(model_dir, 'train')

    for d in train_summary_dir, eval_summary_dir:
        if not os.path.isdir(d):
            os.makedirs(d)
    graph = tf.Graph()
    source = coord.data_source
    batch_size = coord.train_model.batch_size
    max_steps = coord.train_model.max_steps

    with graph.as_default():
        train_writer = tf.summary.FileWriter(train_summary_dir)
        eval_writer = tf.summary.FileWriter(eval_summary_dir)

        train_dataset = source.get_inputs(
            mode=ModeKeys.TRAIN, batch_size=batch_size,
            repeat_count=1)
        eval_dataset = source.get_inputs(
            mode=ModeKeys.EVAL, batch_size=batch_size, repeat_count=1)

        assert(all(isinstance(ds, tf.data.Dataset)
               for ds in (train_dataset, eval_dataset)))

        iterator = tf.data.Iterator.from_structure(
            train_dataset.output_types, train_dataset.output_shapes)

        train_init_op = iterator.make_initializer(train_dataset)
        eval_init_op = iterator.make_initializer(eval_dataset)

        features, labels = iterator.get_next()
        mode = tf.placeholder(shape=(), dtype=tf.string, name='mode')
        train_feed = {mode: ModeKeys.TRAIN}
        eval_feed = {mode: ModeKeys.EVAL}

        spec = coord.get_estimator_spec(features, labels, mode)
        step = tf.train.get_or_create_global_step()
        loss = spec.loss

        saver = tf.train.Saver()
        checkpoint_hook = tf.train.CheckpointSaverHook(
                model_dir, save_secs=save_checkpoints_secs, saver=saver)
        logging_hook = tf.train.LoggingTensorHook(
            dict(loss=loss, step=step), every_n_iter=log_step_count_steps,
            formatter=lambda x:
                'loss at step %d: %s' % (x['step'], str(x['loss'])))

        hooks = [
            checkpoint_hook,
            logging_hook,
            tf.train.NanTensorHook(loss),
        ]

        summary_op = tf.summary.merge_all()
        if summary_op is not None:
            writer = tf.summary.FileWriter(model_dir)
            summary_hook = tf.train.SummarySaverHook(
                save_steps=save_summary_steps,
                summary_writer=writer,
                summary_op=summary_op)
            hooks.append(summary_hook)

        train_op = spec.train_op
        eval_metric_ops = spec.eval_metric_ops
        if 'loss' in eval_metric_ops:
            raise RuntimeError('key "loss" reserved in eval_metric_ops')
        eval_metric_ops['loss'] = tf.metrics.mean(loss)

        keys = sorted(tuple(eval_metric_ops.keys()))
        eval_metric_ops = [eval_metric_ops[k] for k in keys]

        metric_update_ops = [op[1] for op in eval_metric_ops]
        metric_values = [op[0] for op in eval_metric_ops]

        # epoch metrics
        summaries = [
            tf.summary.scalar(k, v, family='epoch_metrics')
            for k, v in zip(keys, metric_values)]
        merged = tf.summary.merge(summaries)

        # reset metric values
        local_init_op = tf.variables_initializer(tf.local_variables())

        # scaffold = tf.train.Scaffold(
        #     local_init_op=(local_init_op, train_init_op))

        # session_creator = tf.train.ChiefSessionCreator(
        #     checkpoint_dir=model_dir, scaffold=scaffold)

        def metrics_loop(sess, feed_dict):
            sess.run(local_init_op)
            while True:
                try:
                    vals, _ = sess.run(
                        (merged, metric_update_ops), feed_dict=feed_dict)
                except tf.errors.OutOfRangeError:
                    break
            return vals

        # with tf.train.MonitoredSession(
        #         session_creator=session_creator, hooks=hooks) as sess:
        with tf.train.SingularMonitoredSession(
                hooks=hooks, checkpoint_dir=model_dir) as sess:
            # sess.run(train_init_op)
            raw_sess = sess.raw_session()
            i = raw_sess.run(step)
            if i >= max_steps:
                print('max steps reached.')
                return

            while i < max_steps:
                for _ in range(epochs_per_eval-1):
                    raw_sess.run(train_init_op)
                    # while not sess.should_stop():
                    while True:
                        try:
                            i, _ = sess.run(
                                (step, train_op), feed_dict=train_feed)
                        except tf.errors.OutOfRangeError:
                            break
                        if i >= max_steps:
                            break

                if i == max_steps:
                    break

                # train and calculate metrics over train set
                raw_sess.run((train_init_op, local_init_op))
                # while not sess.should_stop():
                while True:
                    try:
                        i, vals, _, __ = sess.run(
                            (step, merged, metric_update_ops, train_op),
                            feed_dict=train_feed)
                    except tf.errors.OutOfRangeError:
                        break
                    if i >= max_steps:
                        break
                    # if (log_step_count_steps is not None and
                    #         i % log_step_count_steps == 0):
                    #     print('Loss at step %d: %s' % (i, str(loss_val)))
                # vals = sess.run(merged, feed_dict=train_feed)
                train_writer.add_summary(vals, i)

                # calculate metrics over eval set
                raw_sess.run(eval_init_op)
                vals = metrics_loop(raw_sess, eval_feed)
                eval_writer.add_summary(vals, i)

            raw_sess.run(train_init_op)
            vals = metrics_loop(raw_sess, train_feed)
            train_writer.add_summary(vals, i)
            raw_sess.run(eval_init_op)
            vals = metrics_loop(raw_sess, eval_feed)
            eval_writer.add_summary(vals, i)



def custom_train_and_evaluate2(
        coord, steps=None, max_steps=None,
        save_checkpoints_secs=None, save_checkpoints_steps=None,
        save_summary_steps=100,
        log_step_count_steps=100,
        n_eval_steps=None):
    import tensorflow as tf
    from .listeners import EvalListener
    ModeKeys = tf.estimator.ModeKeys

    if save_checkpoints_secs is None and save_checkpoints_steps is None:
        save_checkpoints_secs = 600

    model_dir = coord.model_dir
    eval_dir = os.path.join(model_dir, 'eval')
    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    graph = tf.Graph()
    with graph.as_default():
        ds = coord.data_source
        batch_size = coord.train_model.batch_size
        train_ds = ds.get_inputs(mode=ModeKeys.TRAIN, batch_size=batch_size)
        eval_ds = ds.get_inputs(mode=ModeKeys.EVAL, batch_size=batch_size)
        if not all(
                isinstance(d, tf.data.Dataset) for d in (train_ds, eval_ds)):
            raise RuntimeError('get_inputs must return a tf.data.Dataset')

        train_iter = train_ds.make_one_shot_iterator()
        eval_iter = eval_ds.make_initializable_iterator()

        mode = tf.placeholder(dtype=tf.string, shape=(), name='mode')
        # handle = tf.cond(
        #     tf.equal(mode, ModeKeys.TRAIN),
        #     train_iter.string_handle,
        #     eval_iter.string_handle
        # )
        handle = tf.placeholder(dtype=tf.string, shape=(), name='data_handle')
        iterator = tf.data.Iterator.from_string_handle(
            handle, train_ds.output_types, train_ds.output_shapes)

        train_handle = train_iter.string_handle()

        features, labels = iterator.get_next()
        spec = coord.get_estimator_spec(features, labels, mode)
        loss = spec.loss
        step = tf.train.get_global_step()

        train_summary = tf.summary.merge_all()

        eval_metric_ops = spec.eval_metric_ops
        if 'loss' in eval_metric_ops:
            raise RuntimeError(
                'Reserved key already present in eval_metric_ops: "loss"')
        eval_metric_ops['loss'] = tf.metrics.mean(loss)

        saver = tf.train.Saver()

        eval_listener = EvalListener(
            eval_iter, eval_metric_ops, mode, handle,
            summary_writer=tf.summary.FileWriter(logdir=eval_dir),
            n_eval_steps=n_eval_steps)
        checkpoint_hook = tf.train.CheckpointSaverHook(
                model_dir, save_secs=save_checkpoints_secs,
                save_steps=save_checkpoints_steps,
                saver=saver, listeners=[eval_listener])
        logging_hook = tf.train.LoggingTensorHook(
            dict(loss=loss, step=step), every_n_iter=log_step_count_steps,
            formatter=lambda x:
                'loss at step %d: %s' % (x['step'], str(x['loss'])))

        hooks = [
            checkpoint_hook,
            logging_hook,
            tf.train.NanTensorHook(loss),
        ]
        if train_summary is not None:
            summary_hook = tf.train.SummarySaverHook(
                save_steps=save_summary_steps,
                summary_writer=tf.summary.FileWriter(logdir=model_dir),
                summary_op=train_summary)
            hooks.append(summary_hook)

        with tf.train.MonitoredSession(hooks=hooks) as sess:
            train_handle = sess.run(train_handle)
            train_feed = {mode: ModeKeys.TRAIN, handle: train_handle}
            i = 0
            train_op = spec.train_op
            while True:
                s, _ = sess.run((step, train_op), train_feed)
                i += 1
                if (steps is not None and i >= steps or
                        max_steps is not None and s >= steps):
                    print('breaking!')
                    break
