from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

FLAGS = flags.FLAGS


flags.DEFINE_float(
    'memory_frac', default=None, help='gpu memory fraction')
flags.DEFINE_string(
    'allow_growth', default=None, help='allow gpu memory growth')
flags.DEFINE_string('action', default=None, help='action to take')

flags.DEFINE_bool(
    'force_confirm', default=False, help='force confirmation for cleaning')

flags.DEFINE_bool(
    'test_variables_changed', default=True,
    help='test if variables are changed during training')
flags.DEFINE_bool(
    'test_update_ops', default=True, help='test if update_ops are run')
flags.DEFINE_bool(
    'use_dummy_inputs', default=False, help='use dummy inputs for profiling')

flags.DEFINE_integer(
    'n_runs', default=10, help='number of runs for tests/profiling')

flags.DEFINE_string('mode', default='train', help='train/eval/infer')
flags.DEFINE_integer(
    'n_eval_steps', default=None, help='number of steps used for evaluation')

flags.DEFINE_string(
    'period', default='00:10:00', help='time per repeat hh:mm:ss')
flags.DEFINE_string(
    'delay', default='00:01:00', help='time before first repeat, hh:mm:ss')


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
    assert(len(g) < 4)
    for g in groups:
        period += g*dt
        dt *= 60
    return period


def get_period():
    return parse_time(FLAGS.period)


def get_delay():
    return parse_time(FLAGS.delay)


def evaluate(coord):
    return coord.evaluate(
        config=get_estimator_config(), steps=FLAGS.n_eval_steps)


def periodic_evaluate(coord):
    import time
    period = get_period()
    delay = get_delay()
    print('Running evaluate periodically')
    print('delay: %ds' % delay)
    print('period: %ds' % period)
    if delay > 0:
        time.sleep(delay)
    while True:
        t = time.time()
        print('Evaluating...')
        evaluate(coord)
        remaining = period - (time.time() - t)
        if remaining > 0:
            print('Sleeping for %ss...' % remaining)
            time.sleep(remaining)


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


def get_estimator_config():
    import tensorflow as tf
    return tf.estimator.RunConfig(session_config=get_session_config())


def vis_inputs(data_source):
    return data_source.vis_inputs(config=get_session_config(), mode=FLAGS.mode)


def report_train_tests(coord):
    return coord.report_train_tests(
        variable_change_test=FLAGS.test_variables_changed,
        update_ops_test=FLAGS.test_update_ops,
        config=get_session_config(),
        steps=FLAGS.n_runs)


def train(coord):
    return coord.train(config=get_estimator_config()),


def train_and_eval(coord):
    import multiprocessing
    memory_frac = FLAGS.memory_frac
    if memory_frac is None:
        raise ValueError('memory_frac must be provided for train_and_eval')
    elif memory_frac > 0.5:
        raise ValueError('memory_frac cannot be greater than 0.5')

    def eval_fn():
        return periodic_evaluate(coord)

    eval_process = multiprocessing.Process(target=eval_fn)
    eval_process.start()
    try:
        train(coord)
    except Exception:
        eval_process.terminate()
        raise
    except KeyboardInterrupt:
        eval_process.terminate()
        raise
    if eval_process.is_alive():
        eval_process.terminate()
    evaluate(coord)


_coord_fns = {
    'vis_inputs': lambda coord: vis_inputs(coord.data_source),
    'train': train,
    'evaluate': evaluate,
    'vis_predictions': lambda coord: coord.vis_predictions(
        config=get_session_config()),
    'profile': lambda coord: coord.create_profile(
        config=get_session_config(), skip_runs=FLAGS.n_runs,
        use_dummy_inputs=FLAGS.use_dummy_inputs),
    'test': lambda coord: report_train_tests(coord),
    'clean': lambda coord: coord.clean(confirm=not FLAGS.force_confirm),
    'periodic_evaluate': periodic_evaluate,
    'train_and_eval': train_and_eval,
}

_coord_fns['eval'] = _coord_fns['evaluate']
_coord_fns['periodic_eval'] = _coord_fns['periodic_evaluate']


def register_coord_fn(action, fn):
    if action in _coord_fns:
        raise KeyError('Action already exists for key "%s"' % action)
    _coord_fns[action] = fn


def coord_main(coord):
    action = FLAGS.action
    return _coord_fns[action](coord)
