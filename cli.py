from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import logging

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS


flags.DEFINE_string('action', default=None, help='action to take')

# RunConfig
flags.DEFINE_integer(
    'keep_checkpoint_max', default=5, help='RunConfig kwarg')
flags.DEFINE_integer(
    'save_checkpoints_steps', default=None, help='RunConfig kwarg')
flags.DEFINE_integer(
    'save_checkpoints_secs', default=None, help='RunConfig kwarg')
flags.DEFINE_integer(
    'save_summary_steps', default=100, help='RunConfig kwarg')
flags.DEFINE_integer(
    'log_step_count_steps', default=100, help='RunConfig kwarg')

# Session config - used in RunConfig
flags.DEFINE_float(
    'memory_frac', default=None, help='gpu memory fraction')
flags.DEFINE_string(
    'allow_growth', default=None, help='allow gpu memory growth')

# for `action == 'clean'`
flags.DEFINE_bool(
    'force_confirm', default=False, help='force confirmation for cleaning')

# for `action == 'test'`
flags.DEFINE_bool(
    'test_variables_changed', default=True,
    help='test if variables are changed during training')
flags.DEFINE_bool(
    'test_update_ops', default=True, help='test if update_ops are run')

# for `action == 'profile'`
flags.DEFINE_bool(
    'use_dummy_inputs', default=False, help='use dummy inputs for profiling')

# for `action in {'test', 'profile'}`
flags.DEFINE_integer(
    'n_runs', default=10, help='number of runs for tests/profiling')

# for `evaluate`/`train_and_evaluate`
flags.DEFINE_integer(
    'n_eval_steps', default=100, help='number of steps used for evaluation')
flags.DEFINE_string(
    'period', default='00:09:00', help='time per repeat hh:mm:ss')
flags.DEFINE_string(
    'delay', default='00:01:00', help='time before first repeat, hh:mm:ss')

# for `action == 'vis_inputs'`
flags.DEFINE_integer(
    'batch_size', default=None, help='batch size for vis_inputs')
flags.DEFINE_string('mode', default=None, help='train/eval/infer')

# for `action == 'count_trainable_parameters'`
flags.DEFINE_list(
    'scope', default=None,
    help='scope name(s) for `count_trainable_parameters`')

flags.DEFINE_string(
    'tf_verbosity', default=None, help='used in tf.logging.set_verbosity')


def get_session_config():
    import tensorflow as tf
    per_process_gpu_memory_fraction = FLAGS.memory_frac
    allow_growth = FLAGS.allow_growth
    if per_process_gpu_memory_fraction is not None:
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
    elif allow_growth is not None:
        gpu_options = tf.GPUOptions(allow_growth=allow_growth)
    else:
        return None

    return tf.ConfigProto(gpu_options=gpu_options)


def get_cl_run_config_kwargs():
    """Get RunConfig kwargs from command line args."""
    kwargs = dict(
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        save_checkpoints_secs=FLAGS.save_checkpoints_secs,
        save_summary_steps=FLAGS.save_summary_steps,
        log_step_count_steps=FLAGS.log_step_count_steps,
        session_config=get_session_config()
    )
    return {k: v for k, v in kwargs.items() if v is not None}


def get_run_config():
    """Get a `tf.estimator.RunConfig` from command line args."""
    import tensorflow as tf
    return tf.estimator.RunConfig(**get_cl_run_config_kwargs())


def parse_time(time_string):
    import re
    regex = r'(?:(\d\d):)?(?:(\d\d):)?(\d\d)'
    prog = re.compile(regex)

    if time_string is None:
        raise ValueError('period must not be None')

    result = prog.match(time_string)

    if result.group(0) != time_string:
        raise ValueError('Invalid time "%s"' % time_string)
    groups = (result.group(i) for i in range(1, 4))
    groups = [int(g) for g in groups if g is not None][-1::-1]
    dt = 1
    period = 0
    assert(len(groups) < 4)
    for g in groups:
        period += g*dt
        dt *= 60
    return period


def get_period():
    return parse_time(FLAGS.period)


def get_delay():
    return parse_time(FLAGS.delay)


def evaluate(coord):
    input_kwargs = {}
    if FLAGS.mode is not None:
        input_kwargs['mode'] = FLAGS.mode
    return coord.evaluate(
        config=get_run_config(),
        input_kwargs=input_kwargs,
        steps=FLAGS.n_eval_steps)


def periodic_evaluate(coord):
    import time
    period = get_period()
    delay = get_delay()
    logging.info('Running evaluate periodically')
    logging.info('delay: %ds' % delay)
    logging.info('period: %ds' % period)
    if delay > 0:
        time.sleep(delay)
    while True:
        t = time.time()
        logging.info('Evaluating...')
        evaluate(coord)
        remaining = period - (time.time() - t)
        if remaining > 0:
            logging.info('Sleeping for %ss...' % remaining)
            time.sleep(remaining)


def vis_inputs(data_source):
    kwargs = dict(config=get_session_config(), batch_size=FLAGS.batch_size)
    mode = FLAGS.mode
    if mode is not None:
        kwargs['mode'] = mode
    return data_source.vis_inputs(**kwargs)


def report_train_tests(coord):
    return coord.report_train_tests(
        variable_change_test=FLAGS.test_variables_changed,
        update_ops_test=FLAGS.test_update_ops,
        config=get_session_config(),
        steps=FLAGS.n_runs)


def train(coord, config_kwargs={}):
    return coord.train(config=get_run_config()),


def train_and_evaluate(coord):
    eval_spec_kwargs = dict(
        throttle_secs=get_period(),
        start_delay_secs=get_delay(),
        steps=FLAGS.n_eval_steps,
    )
    return coord.train_and_evaluate(
        config=get_run_config(), **eval_spec_kwargs)


def count_trainable_variables(coord):
    if FLAGS.mode is not None:
        kwargs = dict(mode=FLAGS.mode)
    else:
        kwargs = {}
    scopes = FLAGS.scope
    scopes = [] if scopes is None else scopes
    kwargs['scope'] = scopes
    total, scope_count = coord.count_trainable_variables(**kwargs)
    for scope, count in zip(scopes, scope_count):
        logging.info('%s: %d' % (scope, count))
    logging.info('Total: %d' % total)


_coord_fns = {
    'vis_inputs': lambda coord: vis_inputs(coord.data_source),
    'train': train,
    'evaluate': evaluate,
    'vis_predictions': lambda coord: coord.vis_predictions(
        config=get_run_config(), data_mode=FLAGS.mode,
        batch_size=FLAGS.batch_size),
    'profile': lambda coord: coord.create_profile(
        config=get_session_config(), skip_runs=FLAGS.n_runs,
        use_dummy_inputs=FLAGS.use_dummy_inputs),
    'profile_inputs': lambda coord: coord.create_inputs_profile(
        config=get_session_config(), skip_runs=FLAGS.n_runs),
    'test': lambda coord: report_train_tests(coord),
    'clean': lambda coord: coord.clean(confirm=not FLAGS.force_confirm),
    'periodic_evaluate': periodic_evaluate,
    'train_and_evaluate': train_and_evaluate,
    'count_trainable_variables': count_trainable_variables,
    'list_trainable_variables':
        lambda coord: coord.list_trainable_variables(scope=FLAGS.scope)
}

_coord_fns['eval'] = _coord_fns['evaluate']
_coord_fns['train_and_eval'] = _coord_fns['train_and_evaluate']
_coord_fns['periodic_eval'] = _coord_fns['periodic_evaluate']
eval = evaluate
periodic_eval = periodic_evaluate
train_and_eval = train_and_evaluate


def register_coord_fn(action, fn):
    if action in _coord_fns:
        raise KeyError('Action already exists for key "%s"' % action)
    _coord_fns[action] = fn


def set_verbosity(tf_verbosity=None):
    import tensorflow as tf
    if tf_verbosity is None:
        tf_verbosity = FLAGS.tf_verbosity
    tf.logging.set_verbosity(getattr(tf.logging, tf_verbosity.upper()))


def coord_main(coord):
    action = FLAGS.action
    if FLAGS.tf_verbosity is not None:
        set_verbosity(FLAGS.tf_verbosity)
    return _coord_fns[action](coord)


def register_experimental():
    from .experimental import custom_train_and_evaluate
    flags.DEFINE_integer(
        'eval_every_secs', default=None,
        help='how often to evaluate in seconds')
    flags.DEFINE_integer(
        'eval_every_steps', default=None,
        help='how often to evaluate in steps')
    def f2(coord):
        kwargs = dict(
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            save_checkpoints_secs=FLAGS.save_checkpoints_secs,
            save_summary_steps=FLAGS.save_summary_steps,
            log_step_count_steps=FLAGS.log_step_count_steps,
            eval_every_secs=FLAGS.eval_every_secs,
            eval_every_steps=FLAGS.eval_every_steps,
            n_eval_steps=FLAGS.n_eval_steps,
        )
        return custom_train_and_evaluate(coord, **kwargs)

    for k in ('custom_train_and_evaluate', 'custom_train_and_eval'):
        # register_coord_fn(k, custom_train_end_evaluate)
        register_coord_fn(k, f2)
