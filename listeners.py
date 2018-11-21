from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .eval_writer import EvalWriter


class EvalListener(tf.train.CheckpointSaverListener):
    def __init__(self, writer):
        self._writer = writer

    def after_save(self, session, global_step_value):
        return self._writer.write(session, global_step_value)

    def end(self, session, global_step_value):
        self._writer.write(session, global_step_value)
